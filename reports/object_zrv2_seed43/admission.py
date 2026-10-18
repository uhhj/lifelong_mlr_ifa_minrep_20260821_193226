"""Read-only parent binding and real CUDA language gate admission; zero updates/env."""
import os,sys,json,hashlib,time,traceback
from pathlib import Path
W=Path('/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018')
O=Path('/root/mlr_ifa/experiments/ZRv2_OBJECT_seed43_20261018')
P=Path('/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1')
S=Path('/data/mlr_ifa/experiment_workspaces/Seed43_three_arm_repeat_20261016_v1/rc/source')
os.environ.update(CORESET_OUT=str(P),CORESET_SEED='43',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
sys.path[:0]=[str(W/'attachment'),str(S)]
import numpy as np
import torch
from coreset_exp.common import sha,write,verify
from coreset_exp.model import cfg,data_setup,complete
from libero.lifelong.utils import control_seed
from reference_core import extend_gates,normalize_rows
from compare_metrics import load_records,metrics,LOCKED

def main():
    tic=time.time();assert not (O/'ADMISSION.json').exists()
    rows=[];binding={};write(O/'RUN_STATUS.json',dict(status='ADMISSION',pid=os.getpid(),time=time.time()))
    try:
        c=verify();rates,records,ep_hashes=load_records(P/'coreset/evaluation',False)
        assert metrics(rates)==LOCKED
        binding.update(parent=str(P),source=str(S),contract_sha256=sha(P/'CONTRACT.json'),modal_sha256=sha(P/'MODAL_CONTRACT.json'),episode_hashes=ep_hashes,metrics={k:str(v) for k,v in metrics(rates).items()},checkpoints={})
        assert binding['modal_sha256']=='c379b9930387abc9eed54e8a9f83394f2f8be80ee1536921a53ce2cab66ea78d'
        inputs={str(P/n):sha(P/n) for n in ['CONTRACT.json','MODAL_CONTRACT.json','FEATURES.json','resolved_eval.json','resolved_coreset.json']}
        for t in range(6,10):
            cp=P/'coreset/models'/f'mlrifa_{t}_model_ep99.pth';h=sha(cp)
            assert h==records[t,t,0]['checkpoint_sha256'];inputs[str(cp)]=h
            binding['checkpoints'][str(t)]=dict(path=str(cp),sha256=h)
            sm=P/'coreset/storage'/f'task{t}.json';m=json.loads(sm.read_text());inputs[str(sm)]=sha(sm)
            assert m['K']==[781,740,780,660][t-6]
            for r in m['selected']:
                assert sha(r['source'])==r['sha256']
                q=P/'coreset/buffer'/r['destination'];assert sha(q)==r['sha256'];inputs[str(q)]=r['sha256']
        inputs.update(c['inputs']);inputs.update({str(S/n):h for n,h in c['source_sha256'].items()})
        fm=json.loads((P/'FEATURES.json').read_text());inputs.update({str(P/'features'/n):h for n,h in fm['hashes'].items()})
        write(O/'PARENT_BINDING.json',binding);write(O/'INPUT_SHA256.json',inputs)
        torch.set_num_threads(4);control_seed(43);torch.backends.cudnn.benchmark=True
        ce=cfg('eval');b,ds,shape=data_setup(ce,False);ce.shape_meta=shape
        p=complete(ce,shape).eval().requires_grad_(False)
        # Tokens for all tasks are metadata only; each gate construction uses its prefix.
        toks=[b.get_task_emb(t) for t in range(10)]
        def encode(ts):
            ids=torch.stack([toks[t][0] for t in ts]).cuda();mask=torch.stack([toks[t][1] for t in ts]).cuda()
            return p.language_encoder(ids,mask).detach().cpu().numpy()
        with torch.no_grad():
            canon=np.concatenate([encode([t]) for t in range(10)])
            variants=[]
            for size in [1,10,32]:
                for t in range(10):
                    for rep in range(3):variants.append((t,f'online_b{size}_r{rep}',encode([t]*size)))
            # Every actual cached language token, including all time positions.
            for t in range(6,10):
                unique={}
                for f in sorted((P/'features'/str(t)).glob('*.pt')):
                    x=torch.load(f,map_location='cpu')['feature'][0,:,0,:].numpy()
                    for v in x:unique.setdefault(v.tobytes(),v)
                variants.append((t,'all_cache_unique',np.stack(list(unique.values()))))
        w=np.empty((0,512));gates={};violations=[]
        for k in range(6,10):
            w,geom=extend_gates(canon[:k],canon[k],w);vr=[]
            for t,label,v in variants:
                if t>k:continue
                a=normalize_rows(v)@w.T;target=np.zeros(len(w));
                if t>=6:target[t-6]=1
                err=np.abs(a-target);off=float(np.max(err[:,target==0])) if np.any(target==0) else 0.
                own=float(np.max(err[:,target==1])) if t>=6 else 0.
                row=dict(task=t,source=label,n=len(v),off=off,own=own,passed=off<=1e-6 and own<=1e-5);vr.append(row)
                if not row['passed']:violations.append(dict(stage=k,**row))
            rows.append(dict(stage=k,geometry=geom,variants=vr));gates[str(k)]=w.copy()
        # Arrays remain server-side, never part of lightweight publication.
        np.savez(W/'language_and_gates.npz',canonical=canon,**{'gate_'+k:v for k,v in gates.items()})
        out=dict(status='PASS' if not violations else 'BLOCKED_GEOMETRY',rows=rows,violations=violations,
                 torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                 numeric_flags=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,deterministic=torch.are_deterministic_algorithms_enabled()),
                 optimizer_calls=0,environment_steps=0,seconds=time.time()-tic)
        write(O/'ADMISSION.json',out);write(O/'RUN_STATUS.json',dict(status=out['status'],time=time.time()))
        print(json.dumps(dict(status=out['status'],violations=violations,seconds=time.time()-tic)),flush=True)
        if violations:raise SystemExit(2)
    except BaseException as e:
        if not (O/'ADMISSION.json').exists():write(O/'ADMISSION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),binding=binding,rows=rows))
        raise
if __name__=='__main__':main()
