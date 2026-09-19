"""One true optimizer update per arm, same original states/windows/RNG."""
import gc,gzip,time,os,traceback
import numpy as np
from scipy.optimize import nnls
from .common import *
from .solver import solve
from paper_active_set_v1.measure import anchor_gradients,mixed_forward,flat,unflatten,project,overlap
from functional_geometry_diagnostic_v1 import geometry as g
from functional_geometry_diagnostic_v1.runtime_guard import Guard
from trajectory_relation_diagnostic_v1 import recovery_gate as recovery
from trajectory_relation_diagnostic_v1.measure import window_nll
from sensitive_component_compatibility_v1.run import probes
from gradient_relation_experiment.preflight import load_banks
torch=g.torch

def norm(x):return float(x.double().norm())
def cosine(x,y):
    den=norm(x)*norm(y)
    return float(x.double()@y.double())/den if den else None

def sample_gradients(algo,Pold,lineage,point,guard):
    ps=g.params(algo);before=g.capture(algo);grads_before=g.digest([None if p.grad is None else p.grad.cpu() for p in ps])
    modes={n:m.training for n,m in algo.policy.named_modules()};algo.policy.eval()
    tasks=sorted({task for half,task in Pold});p=sum(x.numel() for x in ps)
    expected=sum(len(data['feature']) for batches in Pold.values() for data in batches)
    assert expected==32*len(tasks) and expected<=64
    assert p*expected*4<=8*2**30,'Per-sample gradient matrix exceeds fixed8GiB allocation cap'
    H=torch.empty((p,expected),device=ps[0].device,dtype=torch.float32);records=[];col=0
    try:
        for task in tasks:
            for half in ['P0','P1']:
                pos=0
                for data in Pold[half,task]:
                    losses=g.module(algo).policy_head.loss_fn(g.gmm(algo,data),data['actions'].squeeze(1),'none')
                    assert losses.shape==(8,) and torch.isfinite(losses).all()
                    for i in range(8):
                        gg=torch.autograd.grad(losses[i],ps,retain_graph=i<7,allow_unused=True)
                        H[:,col].copy_(flat(gg,ps));row=dict(lineage[point,task,int(half[1:]),pos]);row.update(column=col,initial_nll=float(losses[i].detach()))
                        records.append(row);pos+=1;col+=1;del gg
                    del losses;guard.check()
    finally:
        for n,m in algo.policy.named_modules():m.training=modes[n]
    assert col==expected
    assert all(g.compare_states(before,g.capture(algo)).values()),'Sample differentiation perturbed training state'
    assert grads_before==g.digest([None if p.grad is None else p.grad.cpu() for p in ps])
    del before
    return H,records

def inner(H,u=None,gram=False):
    # Full-dimensional dot products; chunking only controls temporary memory.
    K=torch.zeros((H.shape[1],H.shape[1]),device=H.device,dtype=torch.float64) if gram else None
    f=torch.zeros(H.shape[1],device=H.device,dtype=torch.float64) if u is not None else None
    for start in range(0,H.shape[0],65536):
        B=H[start:start+65536].double()
        if gram:K.add_(B.T@B)
        if u is not None:f.add_(B.T@u[start:start+65536].double())
    return (None if K is None else K.cpu().numpy(),None if f is None else f.cpu().numpy())

def sample_project(H,K,u,independent):
    _,f=inner(H,u);un=norm(u);sol=solve(K,f,un)
    lam=torch.tensor(sol['lam'],device=H.device,dtype=torch.float64);c=torch.empty_like(u)
    for start in range(0,len(u),65536):c[start:start+65536]=(H[start:start+65536].double()@lam).float()
    _,written=inner(H,u-c);hn=np.sqrt(np.maximum(np.diag(K),0));valid=np.array(sol['valid'],dtype=int)
    den=np.maximum(hn*max(un,1e-30),1e-30)
    residual=float(np.maximum(written[valid]/den[valid],0).max()) if len(valid) else 0.
    assert residual<=5e-6,('float32 projected sample QP residual',residual)
    rel=None
    if independent and len(valid):
        # This is an independent solve on full columns, not a check of the same Gram function.
        A=H[:,valid].detach().cpu().double().numpy();target=u.detach().cpu().double().numpy()
        ll=nnls(A,target,maxiter=10000)[0];cc=A@ll
        rel=float(np.linalg.norm(cc-c.detach().cpu().double().numpy())/max(np.linalg.norm(cc),un*1e-12,1e-30))
        assert rel<=2e-6,('full-column NNLS mismatch',rel)
        del A,target,ll,cc
    sol.update(f=f.tolist(),K=K.tolist(),u_norm=un,written_rates=written.tolist(),written_normalized_rate_max=residual,full_parameter_NNLS_relative_error=rel)
    return c,sol

def process(algo,ds,plan,pt,member,origin,witness,lineage,design,protocol,guard,dest):
    begin=time.time();name=pt['point'];seed=pt['seed'];task=pt['task'];ps=g.params(algo)
    torch.cuda.reset_peak_memory_stats();assert sha(origin['anchor']['path'])==origin['anchor']['sha256']
    base=torch.load(origin['anchor']['path'],map_location='cpu');g.restore(algo,base)
    assert all(g.digest(g.capture(algo)[k])==v for k,v in origin['witness'].items())
    native_batch=g.batch(ds,pt['future20_record_indices'][0]);epoch=pt['stage_step']//plan['batches_per_epoch']
    g.native(algo,native_batch,task,epoch);native=g.capture(algo)
    assert all(g.digest(native[k])==v for k,v in witness.items()),('native witness mismatch',name)
    d0=g.displacement(algo,base).cuda();g.restore(algo,base)
    P=probes(algo,ds,member);Pold={k:v for k,v in P.items() if k[1]<task}
    write(dest/'MEMBERSHIP_AUDIT.json',overlap(pt,member,design))
    initialP,_=window_nll(algo,P)
    Htask,tasks,_=anchor_gradients(algo,Pold,fullcheck=True)
    H,sample_rows=sample_gradients(algo,Pold,lineage,name,guard)
    decomposition=[]
    for j,t in enumerate(tasks):
        ix=[r['column'] for r in sample_rows if r['eval_task']==t];assert len(ix)==32
        mean=H[:,ix].mean(dim=1);err=norm(mean-Htask[:,j]);scale=norm(Htask[:,j])
        assert err<=max(1e-6,2e-5*scale),('sample to task gradient mismatch',name,t,err,scale)
        decomposition.append(dict(task=t,absolute=err,relative=err/max(scale,1e-30)))
    K,_=inner(H,gram=True);assert np.isfinite(K).all()
    write(dest/'SAMPLE_GRAM.json',dict(rows=sample_rows,gram=K.tolist(),decomposition=decomposition,parameter_count=H.shape[0]))
    fw=mixed_forward(algo,g.batch(ds,pt['future20_record_indices'][0]));raw=flat(fw['mixed'],ps)
    g.restore(algo,base);native_clip=g.apply_first(algo,fw['mixed'],fw)
    assert all(g.compare_states(native,g.capture(algo)).values()),'Custom native update differs from original'
    states={'MLR':native};candidates={};solutions={};engineering=name in protocol['preflight6']
    for arm in ARMS[1:]:
        guard.check(force=True);u=-raw if 'GRAD' in arm else d0
        if arm.startswith('TASK'):c,sol=project(Htask,u,fullcheck=engineering)
        else:c,sol=sample_project(H,K,u,independent=engineering)
        candidates[arm]=(c,u-c);solutions[arm]=sol
    # All coefficients are decided without reading finite candidate losses or A/B.
    write(dest/'DECISIONS_FROZEN.json',dict(point=name,seed=seed,current_task=task,arms=solutions,sample_reference=sha(dest/'SAMPLE_GRAM.json'),AB_read=False,finite_candidate_loss_used=False))
    decision_hash=sha(dest/'DECISIONS_FROZEN.json');sample_norms=np.sqrt(np.maximum(np.diag(K),0))
    nativeP=None
    for arm in ARMS:
        if arm=='MLR':g.restore(algo,native);clip=native_clip;u=d0;y=d0;c=torch.zeros_like(d0);space='actual_displacement'
        else:
            c,y=candidates[arm];space='raw_descent' if 'GRAD' in arm else 'actual_displacement';u=-raw if 'GRAD' in arm else d0
            if 'GRAD' in arm:
                g.restore(algo,base);clip=g.apply_first(algo,unflatten(raw+c,ps,fw['mixed']),fw)
            else:g.restore(algo,native);g.assign_displacement(algo,native,-c);clip=native_clip
        state=g.capture(algo);actual=g.displacement(algo,base).cuda();states[arm]=state
        for k in ['rng','scheduler','optimizer_step','scheduler_step','training_modes']:assert g.digest(state[k])==g.digest(native[k]),(arm,k)
        if 'STEP' in arm:assert g.digest(state['optimizer'])==g.digest(native['optimizer'])
        if arm.startswith('TASK'):
            old_state=next(x for x in rows(OLD/'measurements'/name/'ENDPOINT_STATES.jsonl') if x['arm']==arm and x['horizon']==1)
            assert all(g.digest(state[k])==v for k,v in old_state['state_hashes'].items()),('historical task QP mismatch',name,arm)
        _,rates0=inner(H,u);_,ratesQ=inner(H,y);_,rates_actual=inner(H,actual)
        actual_norm=norm(actual);d0norm=norm(d0);delta_norm=norm(actual-d0);qnorm=norm(y);unorm=norm(u)
        row=dict(point=name,seed=seed,current_task=task,arm=arm,space=space,rows=len(sample_rows),task_rows=len(tasks),
          trigger=False if arm=='MLR' else bool(any(x>0 for x in solutions[arm]['lam'])),
          target_norm=unorm,projected_norm=qnorm,projected_norm_ratio=qnorm/max(unorm,1e-30),projection_cosine=cosine(u,y),
          requested_correction_norm=norm(c),actual_native_norm=d0norm,actual_norm=actual_norm,actual_norm_ratio=actual_norm/max(d0norm,1e-30),
          actual_cosine_MLR=cosine(actual,d0),actual_correction_norm=delta_norm,actual_correction_ratio=delta_norm/max(d0norm,1e-30),
          near_stall=actual_norm<=.1*d0norm if d0norm>0 else None,
          sample_positive_raw_before=int((rates0>0).sum()),sample_positive_raw_projected=int((ratesQ>0).sum()),sample_positive_raw_actual=int((rates_actual>0).sum()),
          sample_positive_resolvable_actual=int((rates_actual>5e-6*sample_norms*actual_norm+1e-15).sum()),
          postclip_norm=clip['postclip_norm'],preclip_norm=clip['preclip_norm'],**g.moments_distance(state,native))
        emit(dest/'UPDATE_METRICS.jsonl',row)
        for i,s in enumerate(sample_rows):emit(dest/'SAMPLE_RATES.jsonl',dict(s,arm=arm,space=space,gradient_norm=float(sample_norms[i]),target_dot=float(rates0[i]),projected_dot=float(ratesQ[i]),actual_displacement_dot=float(rates_actual[i]),actual_positive_tolerance=float(5e-6*sample_norms[i]*actual_norm+1e-15)))
        vals,_=window_nll(algo,P)
        if arm=='MLR':nativeP=vals
        for (half,t),batches in P.items():
            key=half+'/'+str(t)
            for i,value in enumerate(vals[key]):
                src=lineage[name,t,int(half[1:]),i];initial=initialP[key][i];mlr=nativeP[key][i];eps=64*2**-23*max(1,abs(initial),abs(mlr))
                emit(dest/'PROBE_WINDOWS.jsonl',dict(src,arm=arm,nll=value,initial_nll=initial,MLR_nll=mlr,delta_vs_initial=value-initial,delta_vs_MLR=value-mlr,epsilon=eps))
        write(dest/(arm+'_STATE_HASHES.json'),{k:g.digest(v) for k,v in state.items()})
    assert sha(dest/'DECISIONS_FROZEN.json')==decision_hash
    write(dest/'ALL_UPDATES_FROZEN.json',dict(sha256=decision_hash,AB_read=False,arms=ARMS,time=time.time()))
    # Only now open independent monitor banks; never use them to choose a QP solution.
    g.restore(algo,base);banks=load_banks(algo,design,range(6,task+1));initialAB,baseMeans=window_nll(algo,banks)
    nativeAB=None
    for arm in ARMS:
        g.restore(algo,states[arm]);vals,means=window_nll(algo,banks)
        if arm=='MLR':nativeAB=vals
        for (bank,t),batches in banks.items():
            key=bank+'/'+str(t)
            assert len(vals[key])==len(design['monitor_membership'][str(t)][bank])
            for i,value in enumerate(vals[key]):
                src=design['monitor_membership'][str(t)][bank][i];initial=initialAB[key][i];mlr=nativeAB[key][i];eps=64*2**-23*max(1,abs(initial),abs(mlr))
                emit(dest/'AB_WINDOWS.jsonl',dict(point=name,seed=seed,current_task=task,arm=arm,bank=bank,eval_task=t,window=i,demo=src['demo'],source=src,nll=value,initial_nll=initial,MLR_nll=mlr,delta_vs_initial=value-initial,delta_vs_MLR=value-mlr,epsilon=eps))
            if arm in ['MLR','TASK_GRAD_QP','TASK_STEP_QP']:
                old=next(x for x in rows(OLD/'measurements'/name/'AB_ENDPOINTS.jsonl') if x['arm']==arm and x['horizon']==1 and x['bank']==bank and x['eval_task']==t)
                assert means[key]==old['nll'],('Original A/B task mean changed',name,arm,key)
    assert sha(dest/'DECISIONS_FROZEN.json')==decision_hash
    g.restore(algo,base);assert all(g.compare_states(base,g.capture(algo)).values())
    write(dest/'COMPLETE.json',dict(status='PASS',point=name,seconds=time.time()-begin,native_witness_exact=True,historical_task_QP_first_steps_exact=True,
       sample_task_decomposition=decomposition,full_NNLS_checked=engineering,AB_after_decision_freeze=True,restored=True,
       peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    del states,base,native,H,Htask,candidates,P,Pold,banks,raw,fw;gc.collect();torch.cuda.empty_cache()

def run(mode):
    p=read(OUT/'PROTOCOL_FROZEN.json');assert read(OUT/'ASSET_AUDIT.json')['status']=='CPU_ASSET_AUDIT_PASS'
    assert read(OUT/'UNIT_TESTS.json')['status']=='PASS'
    for n,h in read(OUT/'FROZEN_INPUTS.json').items():assert sha(ROOT/n)==h,n
    if mode=='main':assert read(OUT/'ENGINEERING_PREFLIGHT.json')['status']=='PASS'
    with gzip.open(ROOT/'reports/gradient_relation_4090_v2/DESIGN.json.gz','rt') as f:design=json.load(f)
    pts={x['point']:x for x in design['points']};members={x['point']:x for x in read(ROOT/'reports/sensitive_component_compatibility_v1/PROBE_MEMBERSHIP.json')['points']}
    origins={x['point']:x for s in p['seeds'] for x in rows(GEO/f'carrier/seed{s}/origins.jsonl')}
    witnesses={(x['point'],x['horizon']):x['witness'] for s in p['seeds'] for x in rows(GEO/f'carrier/seed{s}/native_witnesses.jsonl')}
    lineage={(x['point'],x['eval_task'],x['half'],x['position_in_half']):x for x in read(OUT/'PROBE_WINDOW_MEMBERSHIP.json')['windows']}
    torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.cuda.get_device_name(0)==p['resources']['device']
    scratch=Path('/root/sample_task_qp_v1_scratch');scratch.mkdir(exist_ok=True)
    guard=Guard(ROOT,OUT,scratch,{'resource_ceiling':p['resources']},torch);recovery.OUT=OUT;g.init_ddp(OUT/('ddp_'+mode));current=None;algo=None;results=[]
    try:
        for name in p['preflight6'] if mode=='preflight' else p['points']:
            dest=OUT/'measurements'/name
            if (dest/'COMPLETE.json').exists():
                c=read(dest/'COMPLETE.json')
                for n,h in c['files'].items():assert sha(dest/n)==h
                results.append(c);continue
            assert not dest.exists(),('Incomplete point requires audit, no silent retry',name)
            dest.mkdir(parents=True);pt=pts[name]
            if current!=(pt['seed'],pt['task']):
                if algo is not None:del algo,ds;gc.collect();torch.cuda.empty_cache()
                guard.stage('build',name);algo,ds,plan=recovery.build_without_monitors(pt['seed'],pt['task'],design);current=pt['seed'],pt['task']
            guard.stage('measurement',name);process(algo,ds,plan,pt,members[name],origins[name],witnesses[name,1],lineage,design,p,guard,dest)
            results.append(read(dest/'COMPLETE.json'));print('POINT_COMPLETE',mode,name,len(results),flush=True)
        write(OUT/('ENGINEERING_PREFLIGHT.json' if mode=='preflight' else 'MEASUREMENT_COMPLETE.json'),dict(status='PASS',points=len(results),results=results,time=time.time()))
    finally:torch.distributed.destroy_process_group()
if __name__=='__main__':
    import sys
    try:run(sys.argv[1])
    except BaseException:write(OUT/'MEASUREMENT_FAILED.json',dict(error=traceback.format_exc(),time=time.time()));raise
