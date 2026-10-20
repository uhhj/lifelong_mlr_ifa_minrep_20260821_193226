"""Epoch49 old-task collection; exact fixed unique-branch budgets and transactions."""
import os
os.environ["MUJOCO_GL"]="egl"
os.environ["EGL_PLATFORM"]="surfaceless"
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from coreset_exp.common import O, SEED, verify, sha, write, save, rng, restore_rng
from coreset_exp.model import cfg, data_setup, complete, strict_downstream
from libero.lifelong.metric import OffScreenRenderEnv
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed
from . import snapshot_runtime as rt
from .loss import objective


def load(p):
    return json.loads(p.read_text())


@torch.no_grad()
def feature(policy,history,instr,mask,c):
    z=policy.spatial_encode(policy.preprocess_input(rt.batch_of(history,instr,mask,c),train_mode=False))
    assert z.shape==(1,8,4,512)
    return z[0].detach().cpu().clone()


def identity(snap):
    return dict(physical=rt.digest(snap["physical"]),
        observations={k:rt.digest(v) for k,v in snap["obs"].items() if isinstance(v,np.ndarray)})


def branch(env,policy,snap,stage,task,initial,at,phase,rep,mode,instr,mask,c):
    key="s%d_t%d_i%d_%s_r%d_%s"%(stage,task,initial,phase,rep,str(mode))
    commit=rt.O/"committed"/(key+".json")
    payload_path=rt.O/"private"/(key+".pth")
    if commit.exists():
        row=load(commit)
        assert sha(payload_path)==row["payload_sha256"]
        return row,torch.load(payload_path,map_location="cpu")
    obs,history,_=rt.restore(env,policy,snap)
    seed=SEED*100000000+stage*1000000+task*10000+initial*100+rep
    if phase=="confirmation":
        seed+=500000
    table=rt.noise(seed,600-at)
    meta=dict(key=key,stage=stage,task=task,initial=initial,snapshot_step=at,
              phase=phase,repeat=rep,mode=mode,noise_seed=seed,noise_sha256=rt.digest(table))
    rt.ledger("branch_start",**meta)
    windows=[];first_action=None;winner=None;success=False;began=time.monotonic()
    for step in range(600-at):
        history=rt.stack(history,obs)
        par,k=rt.params(policy,rt.batch_of(history,instr,mask,c))
        action=rt.normal(par,k,table[step])
        if step==0:
            winner=k
            if mode!="normal":
                action=par[0][int(mode)].copy()
            first_action=action.copy()
        assert np.isfinite(action).all()
        if step<8:
            windows.append(dict(feature=feature(policy,history,instr,mask,c),
                action=torch.from_numpy(action.copy()),weight=1. if step==0 else .25,
                task=task,initial=initial,branch_key=key,continuation_step=step))
        obs,success=rt.envstep(env,action,"screening")
        if success:
            break
    save(payload_path,windows)
    row=dict(**meta,success=bool(success),steps=step+1,seconds=time.monotonic()-began,
        first_action_sha256=rt.digest(first_action),winner=winner,
        payload_sha256=sha(payload_path),status="COMMITTED")
    write(commit,row)
    rt.ledger("branch_complete",**row)
    rt.COUNTS["completed_branches"]+=1
    rt.status("COLLECTING",key=key)
    return row,windows


def counts(base,candidate):
    return (sum(not a[0]["success"] and b[0]["success"] for a,b in zip(base,candidate)),
            sum(a[0]["success"] and not b[0]["success"] for a,b in zip(base,candidate)))


def qualifies(base,result):
    fs,sf=counts(base,result)
    return fs>=2 and sf==0


def certify(env,policy,snap,stage,task,initial,at,instr,mask,c):
    def run(phase,rep,mode):
        return branch(env,policy,snap,stage,task,initial,at,phase,rep,mode,instr,mask,c)
    normal=[run("exploration",r,"normal") for r in range(3)]
    winner=normal[0][0]["winner"]
    results={m:[run("exploration",r,m) for r in range(3)] for m in range(5)}
    eligible=[]
    for m in range(5):
        if m==winner:
            continue
        child=results[m]
        if qualifies(normal,child) and qualifies(results[winner],child):
            if all(child[r][0]["first_action_sha256"] not in (
                    normal[r][0]["first_action_sha256"],results[winner][r][0]["first_action_sha256"])
                   for r in range(3)):
                eligible.append((-(counts(normal,child)[0]+counts(results[winner],child)[0]),m))
    if not eligible:
        return [],dict(status="NO_EXPLORATION_CANDIDATE",winner=winner)
    _,chosen=min(eligible)
    nc=[run("confirmation",r,"normal") for r in range(3)]
    mc=[run("confirmation",r,winner) for r in range(3)]
    cc=[run("confirmation",r,chosen) for r in range(3)]
    metadata=dict(winner=winner,chosen=chosen,exploration_vs_normal=counts(normal,results[chosen]),
        exploration_vs_argmax=counts(results[winner],results[chosen]),
        confirmation_vs_normal=counts(nc,cc),confirmation_vs_argmax=counts(mc,cc))
    if not (qualifies(nc,cc) and qualifies(mc,cc)):
        return [],dict(status="CONFIRMATION_FAILED_NO_REPLACEMENT",**metadata)
    if not all(cc[r][0]["first_action_sha256"] not in (
            nc[r][0]["first_action_sha256"],mc[r][0]["first_action_sha256"]) for r in range(3)):
        return [],dict(status="CONFIRMATION_ACTION_NOT_DISTINCT",**metadata)
    repeat=next(r for r in range(3) if cc[r][0]["success"]
                and not nc[r][0]["success"] and not mc[r][0]["success"])
    # One fixed successful confirmation continuation, never all successful branches.
    selected=cc[repeat][1]
    assert len(selected)<=8 and selected[0]["weight"]==1.
    return selected,dict(status="CERTIFIED",selected_confirmation_repeat=repeat,
        training_branch_key=cc[repeat][0]["key"],windows=len(selected),**metadata)


def gradient_gate(records,cp,c):
    if not records:
        return dict(status="NO_CERTIFIED_RECORDS",optimizer_calls=0)
    state=rng()
    try:
        p=BriefPolicy(cfg("ra_coreset"),None).cuda().eval()
        strict_downstream(p,cp)
        value,_=objective(p,records[:4],torch.device("cuda"))
        value.backward()
        grads=[x.grad for x in p.parameters() if x.grad is not None]
        assert grads and all(torch.isfinite(x).all() for x in grads)
        norm=sum(float(x.abs().sum()) for x in grads)
        assert norm>0
        return dict(status="PASS",gradient_l1=norm,optimizer_calls=0)
    finally:
        restore_rng(state)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",type=int,choices=(7,8,9))
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    verify()
    root=O/"covr"/("stage%d"%args.stage)
    if (root/"ACCEPTANCE.json").exists():
        assert load(root/"ACCEPTANCE.json")["status"]=="PASS"
        return
    assert args.resume or not root.exists(),"Explicit technical resume required"
    root.mkdir(parents=True,exist_ok=True)
    rt.initialize(root)
    cp=O/"ra_coreset/midstage"/("mlrifa_%d_model_ep49.pth"%args.stage)
    cp_hash=sha(cp)
    c=cfg("eval")
    control_seed(SEED);torch.backends.cudnn.benchmark=True
    bench,_,shape=data_setup(c,False);c.shape_meta=shape
    p=complete(c,shape);strict_downstream(p,cp);p.eval().requires_grad_(False)
    decisions=load(root/"DECISIONS.json") if (root/"DECISIONS.json").exists() else {}
    for task_id in range(6,args.stage):
        task=bench.get_task(task_id);instr,mask=bench.get_task_emb(task_id)
        states=torch.load(Path(c.init_states_folder)/task.problem_folder/task.init_states_file)
        assert len(states)>=32
        selected=sum(v["selected"] for k,v in decisions.items() if k.startswith("t%d_"%task_id))
        for initial in range(20,26):
            key="t%d_i%d"%(task_id,initial)
            if key in decisions:
                continue
            at=(40,120,240,40,120,240)[initial-20]
            normalfile=root/"normal"/(key+".json")
            previous=load(normalfile) if normalfile.exists() else None
            if previous and (previous["success"] or not previous["snapshot_available"] or selected>=3):
                decisions[key]=dict(selected=False,reason="NOT_ELIGIBLE",normal=previous)
                write(root/"DECISIONS.json",decisions)
                continue
            env=OffScreenRenderEnv(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
                camera_heights=c.data.img_h,camera_widths=c.data.img_w)
            try:
                env.seed(SEED);env.reset();env.seed(SEED);p.reset()
                obs=env.set_init_state(states[initial]);history=[];snap=None
                rt.ledger("normal_start" if previous is None else "technical_prefix_recollection_start",
                    stage=args.stage,task=task_id,initial=initial)
                for _ in range(5):
                    obs,_=rt.envstep(env,np.zeros(7),"prefix")
                table=rt.noise(850000000+SEED*100000+args.stage*10000+task_id*100+initial,600)
                success=False
                for t in range(at if previous else 600):
                    history=rt.stack(history,obs)
                    par,k=rt.params(p,rt.batch_of(history,instr,mask,c))
                    obs,success=rt.envstep(env,rt.normal(par,k,table[t]),"prefix")
                    if t+1==at and not success:
                        snap=rt.snapshot(env,p,obs,history[-7:])
                    if success:
                        break
                if previous:
                    assert not success and snap is not None
                    assert identity(snap)==previous["snapshot_identity"]
                    normal=previous
                    rt.ledger("technical_prefix_recollection_complete",stage=args.stage,task=task_id,initial=initial)
                else:
                    normal=dict(stage=args.stage,task=task_id,initial=initial,success=bool(success),
                        steps=t+1,noise_sha256=rt.digest(table),initial_sha256=rt.digest(states[initial]),
                        snapshot_available=snap is not None,
                        snapshot_identity=identity(snap) if snap is not None else None)
                    write(normalfile,normal)
                    rt.ledger("normal_complete",**normal)
                if normal["success"] or snap is None or selected>=3:
                    decisions[key]=dict(selected=False,reason="NOT_ELIGIBLE",normal=normal)
                else:
                    gate=rt.gate(env,p,snap,initial,instr,mask,c)
                    write(root/("gate_%s_%d.json"%(key,time.time_ns())),gate)
                    windows,certificate=certify(env,p,snap,args.stage,task_id,initial,at,instr,mask,c)
                    save(root/"selected"/(key+".pth"),windows)
                    decisions[key]=dict(selected=True,normal=normal,certificate=certificate,
                        payload_sha256=sha(root/"selected"/(key+".pth")))
                    selected+=1
                write(root/"DECISIONS.json",decisions)
            finally:
                env.close()
    records=[]
    for key in sorted(decisions):
        d=decisions[key]
        if d["selected"]:
            path=root/"selected"/(key+".pth")
            assert sha(path)==d["payload_sha256"]
            records.extend(torch.load(path,map_location="cpu"))
    allbranches=[load(p) for p in (root/"committed").glob("*.json")]
    cells=args.stage-6
    assert len(decisions)==6*cells
    assert sum(d["selected"] for d in decisions.values())<=3*cells
    assert sum(r["phase"]=="exploration" for r in allbranches)<=54*cells
    assert sum(r["phase"]=="confirmation" for r in allbranches)<=27*cells
    assert all(r["steps"]<=600-r["snapshot_step"] for r in allbranches)
    save(root/"sidecar.pth",records)
    gradient=gradient_gate(records,cp,c)
    assert sha(cp)==cp_hash
    verify()
    write(root/"ACCEPTANCE.json",dict(status="PASS",stage=args.stage,normal_episodes=6*cells,
        selected_snapshots=sum(d["selected"] for d in decisions.values()),
        certified_snapshots=sum(d.get("certificate",{}).get("status")=="CERTIFIED" for d in decisions.values()),
        unique_branches=len(allbranches),training_windows=len(records),sidecar_sha256=sha(root/"sidecar.pth"),
        sidecar_bytes=(root/"sidecar.pth").stat().st_size,real_gradient=gradient,
        invocation_counts=rt.COUNTS,checkpoint_sha256=cp_hash,optimizer_calls=0))


if __name__=="__main__":
    main()
