"""Exactly ten fresh training-state episodes at acquisition; no evaluation-state replay."""
import argparse,collections,contextlib,io
from runtime import *
from memory import tensor_sha
from evaluate import encode,load,ah
from coreset_exp.model import data_setup,complete
from libero.lifelong.metric import raw_obs_to_tensor_obs,SubprocVectorEnv,OffScreenRenderEnv

def event(kind,**kw):
    with (R/'COLLECTION_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kw))+'\n');f.flush()

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=list(ARMS));ap.add_argument('task',type=int);args=ap.parse_args();arm=args.arm;taskid=args.task
    assert taskid in range(6,10);verify();capacity()
    key='%s_%d'%(arm,taskid);assert not (R/('COLLECTION_STARTED_'+key+'.json')).exists(),'No blind collection retry'
    c=EasyDict(js(O/'resolved_eval.json'));c.training=False;control_seed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):b,_,shape=data_setup(c,False)
    c.shape_meta=shape;front=complete(c,shape).eval().requires_grad_(False);policy=load(arm,taskid,c,shape)
    cp=sha(model_path(arm,taskid));instr,mask=b.get_task_emb(taskid);task=b.get_task(taskid)
    path=Path(c.init_states_folder)/task.problem_folder/task.init_states_file;initials=torch.load(path,map_location='cpu')
    split=js(R/'INITIAL_SPLIT.json')[str(taskid)];assert sha(path)==split['file_sha256']
    indices=list(range(20,30));assert all(ah(np.asarray(initials[i]))==split['training'][str(i)] for i in indices)
    # Actual length 1..8 cache parity under this checkpoint, zero environment steps.
    gen=torch.Generator().manual_seed(20261025);obs={k:torch.rand((2,8,*v),generator=gen).cuda() for k,v in shape['all_shapes'].items()}
    base=dict(task_emb=instr.repeat(2,1).cuda(),masks=mask.repeat(2,1).cuda())
    frames=[encode(front,dict(obs={k:v[:,i:i+1] for k,v in obs.items()},**base)) for i in range(8)];parity=[]
    for length in range(1,9):
        whole=encode(front,dict(obs={k:v[:,:length] for k,v in obs.items()},**base));cached=torch.cat(frames[:length],1)
        q0=policy.policy_head.forward_fn(policy.temporal_encode(whole)[:,-1]);q1=policy.policy_head.forward_fn(policy.temporal_encode(cached)[:,-1])
        assert torch.allclose(whole,cached,atol=1e-4,rtol=1e-4) and all(torch.allclose(x,y,atol=1e-4,rtol=1e-4) for x,y in zip(q0,q1))
        parity.append(dict(length=length,feature_max_abs=float((whole-cached).abs().max()),parameter_max_abs=max(float((x-y).abs().max()) for x,y in zip(q0,q1))))
    write(R/('COLLECTION_ADMISSION_'+key+'.json'),dict(status='PASS',cp_sha256=cp,cache_parity=parity,frontend_calls=16,policy_calls=16,updates=0,environments=0))
    write(R/('COLLECTION_STARTED_'+key+'.json'),dict(time=time.time(),pid=os.getpid(),attempt_budget=10,cp_sha256=cp))
    args_env=dict(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),camera_heights=c.data.img_h,camera_widths=c.data.img_w)
    eps=np.stack([np.random.RandomState(20271025+SEED*1000+taskid*100+i).standard_normal((600,7)).astype(np.float32) for i in indices])
    quotas=[MEMORY_PER_TASK//10+int(j<MEMORY_PER_TASK%10) for j in range(10)]
    reservoirs=[[] for _ in indices];generators=[np.random.RandomState((90100+SEED)+taskid*100+i) for i in indices]
    seen=[0]*10;hashers=[hashlib.sha256() for _ in indices];tic=time.monotonic();calls=0;parameter_calls=0
    event('start',arm=arm,task=taskid,cp_sha256=cp,initial_indices=indices)
    env=SubprocVectorEnv([lambda:OffScreenRenderEnv(**args_env) for _ in indices])
    try:
        control_seed(SEED);env.seed(SEED+100);env.reset();env.seed(SEED+100);ob=env.set_init_state(initials[indices])
        for _ in range(5):ob,_,_,_=env.step(np.zeros((10,7)))
        hist=collections.deque(maxlen=8);done=np.zeros(10,dtype=bool);end=np.full(10,600)
        for step in range(600):
            data=raw_obs_to_tensor_obs(ob,instr,mask,c);data['obs']={k:v[:,None] for k,v in data['obs'].items()}
            hist.append(encode(front,data));calls+=1;x=torch.cat(list(hist),1)
            mu,sd,z=policy.policy_head.forward_fn(policy.temporal_encode(x)[:,-1]);parameter_calls+=1
            mode=z.argmax(-1);ix=torch.arange(10,device='cuda');act=(mu[ix,mode]+sd[ix,mode]*torch.from_numpy(eps[:,step]).cuda()).cpu().numpy();act[done]=0
            for j in range(10):
                if done[j]:continue
                hashers[j].update(np.ascontiguousarray(act[j]).tobytes());seen[j]+=1
                if arm!='D':
                    n=seen[j];slot=n-1 if n<=quotas[j] else int(generators[j].randint(n))
                    if slot<quotas[j]:
                        row=dict(feature=x[j].detach().cpu().clone(),mu=mu[j,mode[j]].detach().cpu().clone(),sd=sd[j,mode[j]].detach().cpu().clone(),task=taskid,trajectory=j,initial_index=indices[j],step=step,length=len(hist),mode=int(mode[j]),source='online')
                        if n<=quotas[j]:reservoirs[j].append(row)
                        else:reservoirs[j][slot]=row
            ob,_,success,_=env.step(act)
            for j in range(10):
                if not done[j] and success[j]:done[j]=True;end[j]=step+1
            if done.all():break
        outcomes=[dict(arm=arm,task=taskid,initial_index=i,initial_sha256=ah(np.asarray(initials[i])),initial_file_sha256=sha(path),noise_sha256=ah(eps[j]),action_stream_sha256=hashers[j].hexdigest(),cp_sha256=cp,success=bool(done[j]),steps=int(end[j]),env_seed=SEED+100,attempt=1,termination='success' if done[j] else 'timeout600',purpose='training_collection_not_formal_evaluation') for j,i in enumerate(indices)]
        assert sum(seen)==sum(int(r['steps']) for r in outcomes)
        write(R/('COLLECTION_OUTCOMES_'+key+'.json'),outcomes)
    finally:env.close()
    ds=Dataset(taskid);valid=[i for i,r in enumerate(ds.rows) if r['task']==taskid and r['bc_valid']]
    y=torch.stack([torch.as_tensor(ds.rows[i]['next_action']) for i in valid]).float()
    scales=torch.stack([y[:,lo:hi].square().mean().sqrt().clamp_min(.1) for lo,hi in [(0,3),(3,6),(6,7)]])
    if arm=='D':
        demos=sorted({ds.rows[i]['demo'] for i in valid});rows=[];extra_calls=0
        for j,demo in enumerate(demos):
            ids=sorted([i for i in valid if ds.rows[i]['demo']==demo],key=lambda i:ds.rows[i]['u']);quota=MEMORY_PER_TASK//len(demos)+int(j<MEMORY_PER_TASK%len(demos))
            chosen=np.random.RandomState((90100+SEED)+taskid*100+j).choice(len(ids),size=min(quota,len(ids)),replace=False)
            for pos in sorted(chosen):
                i=ids[int(pos)];feature=ds.native[i][0]['feature'].squeeze(0).float();q=params(policy,feature[None].cuda());extra_calls+=1;k=int(q[2][0].argmax())
                rows.append(dict(feature=feature.clone(),mu=q[0][0,k].cpu(),sd=q[1][0,k].cpu(),task=taskid,trajectory=j,demo=demo,u=ds.rows[i]['u'],source_index=i,source_sha256=ds.rows[i]['sample_sha256'],length=len(feature),mode=k,source='demonstration'))
    else:
        rows=sorted([row for j,rr in enumerate(reservoirs) if arm=='O' or done[j] for row in rr],key=lambda r:(r['trajectory'],r['step']));extra_calls=0
    assert len(rows)<=MEMORY_PER_TASK and sum(len(torch.load(memory_path(arm,t),map_location='cpu')['rows']) for t in range(6,taskid))+len(rows)<=512
    # One immutable 128-window allocation per task: no retired histories are re-read or archived beyond the cap.
    p=memory_path(arm,taskid);assert not p.exists()
    save(p,dict(rows=rows,scales=scales,source_sha256=cp,task=taskid,arm=arm,contract=verify()))
    meta=[]
    for row in rows:
        public={k:v for k,v in row.items() if k not in ['feature','mu','sd']}
        public.update(feature_sha256=tensor_sha(row['feature']),mu_sha256=tensor_sha(row['mu']),sd_sha256=tensor_sha(row['sd']))
        meta.append(public)
    write(R/('MEMORY_'+key+'.json'),dict(status='PASS_FROZEN_ACQUISITION_MEMORY',arm=arm,task=taskid,count=len(rows),limit=MEMORY_PER_TASK,rows=meta,scales=scales.tolist(),source_sha256=cp,cache_sha256=sha(p),collection_outcomes_sha256=sha(R/('COLLECTION_OUTCOMES_'+key+'.json')),collection_successes=int(done.sum()),training_attempts=10,online_frontend_calls=calls,online_policy_calls=parameter_calls,demonstration_policy_calls=extra_calls,parity_frontend_calls=16,parity_policy_calls=16,seconds=time.monotonic()-tic,optimizer_updates=0,terminal_task_unused_for_future_training=taskid==9))
    event('complete',arm=arm,task=taskid,seconds=time.monotonic()-tic,attempts=10,successes=int(done.sum()),memory_windows=len(rows));verify()

if __name__=='__main__':main()
