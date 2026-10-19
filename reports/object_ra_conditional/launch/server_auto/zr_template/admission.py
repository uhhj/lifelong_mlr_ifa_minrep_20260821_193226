"""Read-only parent binding and real CUDA language gate admission; zero updates/env."""
import os,sys,json,hashlib,time,traceback
from pathlib import Path
from zr2_exp.params import *
os.environ.update(CORESET_OUT=str(P),CORESET_SEED=str(SEED),PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
sys.path[:0]=[str(W/'attachment'),str(S)]
import numpy as np
import torch
from coreset_exp.common import sha,write,verify
from coreset_exp.model import cfg,data_setup,complete
from libero.lifelong.utils import control_seed
from reference_core import extend_gates,normalize_rows
from branch_metrics import load_records,metrics

def main():
    tic=time.time();assert not (O/'ADMISSION.json').exists()
    rows=[];binding={};write(O/'RUN_STATUS.json',dict(status='ADMISSION',pid=os.getpid(),time=time.time()))
    try:
        c=verify();rates,records,ep_hashes=load_records(P/PARM/'evaluation',False)
        assert SPEC['parent_engineering_file'] and json.loads(Path(SPEC['parent_engineering_file']).read_text())['status']=='PASS'
        binding.update(parent=str(P),parent_arm=PARM,seed=SEED,suite=SUITE,source=str(S),contract_sha256=sha(P/'CONTRACT.json'),modal_sha256=sha(P/'MODAL_CONTRACT.json'),episode_hashes=ep_hashes,metrics={k:str(v) for k,v in metrics(rates).items()},checkpoints={})
        assert cfg(PARM).benchmark_name==SUITE and cfg(PARM).seed==SEED
        assert all(cfg(PARM).policy[k] is True for k in ('use_vision','use_language','use_state'))
        assert cfg(PARM).gamma==(.3 if SUITE=='LIBERO_OBJECT' else .7)
        inputs={str(P/n):sha(P/n) for n in ['CONTRACT.json','MODAL_CONTRACT.json','FEATURES.json','resolved_eval.json','resolved_'+PARM+'.json']}
        for t in range(6,10):
            cp=P/PARM/'models'/f'mlrifa_{t}_model_ep99.pth';h=sha(cp)
            assert h==records[t,t,0]['checkpoint_sha256'];inputs[str(cp)]=h
            binding['checkpoints'][str(t)]=dict(path=str(cp),sha256=h)
            sm=P/PARM/'storage'/f'task{t}.json';m=json.loads(sm.read_text());inputs[str(sm)]=sha(sm)
            assert m['K']==QUOTAS[t-6]
            for r in m['selected']:
                assert sha(r['source'])==r['sha256']
                q=P/PARM/'buffer'/r['destination'];assert sha(q)==r['sha256'];inputs[str(q)]=r['sha256']
        inputs.update(c.get('inputs',{}));inputs.update(c.get('files',{}));inputs[SPEC['parent_engineering_file']]=sha(SPEC['parent_engineering_file']);inputs.update({str(S/n):h for n,h in c['source_sha256'].items()})
        fm=json.loads((P/'FEATURES.json').read_text());assert [fm['counts'][str(t)] for t in range(6,10)]==COUNTS;inputs.update({str(P/'features'/n):h for n,h in fm['hashes'].items()})
        inputs.update({str(P/PARM/'evaluation'/n):h for n,h in ep_hashes.items()})
        for extra in SPEC.get('parent_amendment_files',[]):inputs[extra]=sha(extra)
        write(O/'PARENT_BINDING.json',binding);write(O/'INPUT_SHA256.json',inputs)
        torch.set_num_threads(4);control_seed(SEED);torch.backends.cudnn.benchmark=True
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
