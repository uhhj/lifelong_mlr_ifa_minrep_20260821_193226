"""Measure C0 and original-trajectory prefixes, never open outcome labels."""
import argparse
import gc
import math
import os
import time
import numpy as np
from .common import *
from .numeric import avg,tail,linear_error,mmd,conditional,standardize

def subset(d,indices):
    return dict(d,feature=d['feature'][indices],actions=d['actions'][indices],rows=[d['rows'][i] for i in indices])

def halves(d):
    split=read(OUT/'DEMO_SPLITS.json')[str(d['task'])]
    return np.array([0 if r['trajectory_id'] in split['D1'] else 1 for r in d['rows']])

def ids(d):return np.array([r['trajectory_id'] for r in d['rows']])

def extract(algo,d):
    import torch
    from task_relation_forecast_v1 import runtime as rt
    from gradient_relation_experiment import kernel as g
    with rt.evaluation(algo),torch.no_grad():
        seq=torch.cat([g.module(algo).temporal_encode(x.cuda()) for x in d['feature'].split(16)])
        z=seq[:,-1,:]
        pooled=torch.stack([seq[i,:r['valid_frame_count']].mean(0) for i,r in enumerate(d['rows'])])
        head=g.module(algo).policy_head
        means,stds,logits=head.forward_fn(z)
        expected=(logits.softmax(-1)[:,:,None]*means).sum(1)
        nll=head.loss_fn(head(z),d['actions'].cuda(),reduction='none')
    return dict(z=pooled.cpu().numpy().astype('float64'),expected=expected.cpu().numpy().astype('float64'),nll=nll.cpu().numpy().astype('float64'))

def matching(d):
    import h5py
    entry=read(OLD/'DATA_MANIFEST.json')['tasks'][str(d['task'])];values=[];switch=[]
    with h5py.File(entry['path'],'r') as f:
        for r in d['rows']:
            demo=f['data/'+r['trajectory_id']];start=r['window_index'];end=min(start+7,r['trajectory_length']-1)
            values.append(np.concatenate([demo['obs/'+k][end].reshape(-1) for k in ('joint_states','gripper_states','ee_pos','ee_ori')]))
            action=demo['actions'][max(0,start-1):end+1,-1]
            switch.append(bool(np.any(np.diff(np.sign(action))!=0)))
    meta=read(OUT/'matching'/('task%d.json'%d['task']));assert meta['rows']==d['rows']
    path=STORE/'matching'/('task%d.npz'%d['task']);assert sha(path)==meta['sha256']
    with np.load(path) as z:vision=z['visual'].copy()
    return np.concatenate([vision,np.asarray(values)],axis=1),np.array(switch)

def gradient(algo,d,index):
    import torch
    from task_relation_forecast_v1 import runtime as rt
    from gradient_relation_experiment import kernel as g
    params=tuple(p for p in algo.policy.parameters() if p.requires_grad)
    with rt.evaluation(algo):
        model=g.module(algo);feature=d['feature'][index].cuda();actions=d['actions'][index].cuda()
        z=model.temporal_encode(feature)[:,-1,:]
        loss=model.policy_head.loss_fn(model.policy_head(z),actions,reduction='none')
        idv=ids(d)[index];weights=np.array([1/(len(set(idv))*sum(idv==q)) for q in idv])
        objective=(loss*torch.tensor(weights,device=loss.device,dtype=loss.dtype)).sum()
        gg=torch.autograd.grad(objective,params,allow_unused=True)
        return tuple(torch.zeros_like(p).cpu() if v is None else v.detach().cpu() for p,v in zip(params,gg))

def mean_optional(values):return None if any(v is None for v in values) else float(np.mean(values))

def static(k,s,t,repeat_run=False):
    import torch
    from task_relation_forecast_v1 import runtime as rt
    from task_relation_forecast_v1.signals import gn_basis
    from gradient_relation_experiment import kernel as g
    from libero.lifelong.algos.mlr import _pairwise_cosine_mean
    from .vendor.LogME import LogME
    dest=OUT/('static_repeat' if repeat_run else 'static')/unit(k,s,t);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    begin=time.monotonic();sd=rt.load_data(k,s,'D_signal','signal',s,t);td=rt.load_data(k,t,'D_signal','signal',s,t)
    algo=rt.build(k,t,STORE/'bootstrap'/unit(k,s,t));hs,ht=halves(sd),halves(td)
    try:
        initial=g.capture(algo);rt.load_model(algo,rt.model_path(k,'base'));bs=extract(algo,sd);bt=extract(algo,td)
        rt.load_model(algo,rt.model_path(k,'source',source=s,step=1000));ss=extract(algo,sd);st=extract(algo,td)
        repeat=extract(algo,td)
        assert all(np.isfinite(v).all() for obj in (bs,bt,ss,st) for v in obj.values())
        witness=g.capture(algo)
        costs={};torch.cuda.synchronize();clock=time.monotonic()
        xs,switchs=matching(sd);xt,switcht=matching(td)
        matching_seconds=time.monotonic()-clock;clock=time.monotonic()
        action=td['actions'][:,-1,:].numpy().astype('float64');asource=sd['actions'][:,-1,:].numpy().astype('float64')
        scale=np.array(read(OUT/'ACTION_SCALES.json')[str(t)]['std'])
        common_scale=np.sqrt((np.array(read(OUT/'ACTION_SCALES.json')[str(s)]['std'])**2+scale**2)/2)
        affinity=bt['nll']-st['nll'];nll_delta_repeat=repeat['nll']-st['nll']
        ebase=((bt['expected']-action)/scale)**2;esource=((st['expected']-action)/scale)**2
        half_results=[{},{}];rawdiag={}
        for half in (0,1):
            si=np.where(hs==half)[0];ti=np.where(ht==half)[0];ds=subset(sd,si);dt=subset(td,ti)
            q=half_results[half]
            q['visual']=float(_pairwise_cosine_mean(ds['feature'][:,:,1,:].reshape(-1,512),dt['feature'][:,:,1,:].reshape(-1,512)))
            q['language']=float(_pairwise_cosine_mean(ds['feature'][:,0,0,:],dt['feature'][:,0,0,:]))
            q['affinity']=avg(affinity[ti],ids(td)[ti]);q['affinity_lower_tail']=tail(affinity[ti],ids(td)[ti],False)
            q['action_improvement']=avg((ebase-esource)[ti,:6].mean(1),ids(td)[ti]);q['gripper_improvement']=avg((ebase-esource)[ti,6],ids(td)[ti])
            phase=np.array([min(2,int(3*r['window_index']/r['trajectory_length'])) for r in td['rows']])
            q['affinity_worst_third']=min(avg(affinity[ti[phase[ti]==a]],ids(td)[ti[phase[ti]==a]]) for a in range(3) if np.any(phase[ti]==a))
            selected=ti[switcht[ti]];q['affinity_gripper_switch']=avg(affinity[selected],ids(td)[selected]) if len(selected) else None
            q['b_target']=avg(bt['nll'][ti],ids(td)[ti]);q['b_source']=avg(ss['nll'][si],ids(sd)[si]);q['b_progress']=avg((bs['nll']-ss['nll'])[si],ids(sd)[si])
        torch.cuda.synchronize();cheap_bundle=time.monotonic()-clock
        for name in ('S01','S04','S05','S06'):costs[name]=cheap_bundle
        for field,xx in [('linear_probe',(bt['z'],st['z'])),('label_mse',(bt['expected'],st['expected']))]:
            clock=time.monotonic()
            eb=linear_error(xx[0],action,ht,ids(td));es=linear_error(xx[1],action,ht,ids(td))
            for h in (0,1):half_results[h][field]=(eb[h]-es[h])/(eb[h]+1e-8)
            rawdiag[field]=dict(base_errors=eb,source_errors=es)
            costs['S07' if field=='linear_probe' else 'S08']=time.monotonic()-clock
        clock=time.monotonic()
        for h,value in enumerate(mmd(ss['z'],st['z'],hs,ht)):half_results[h]['mmd']=value
        costs['S02']=time.monotonic()-clock;clock=time.monotonic()
        for h,values in enumerate(conditional(xs,xt,asource,action,hs,ht,common_scale)):
            half_results[h].update(zip(('coverage','conditional_action','conditional_gripper'),values))
        matching_amortized=sum(read(OUT/'matching'/('task%d.json'%task))['seconds'] for task in (s,t))/18
        costs['S10']=time.monotonic()-clock+matching_seconds+matching_amortized;clock=time.monotonic()
        for h in (0,1):
            tr=ht==h;te=~tr;logs=[]
            _,yy=standardize(action[tr],action[te],.05)
            for zz in (bt['z'],st['z']):
                _,zztest=standardize(zz[tr],zz[te])
                score=float(LogME(regression=True).fit(zztest,yy));logs.append(score)
            half_results[h]['logme_difference']=logs[1]-logs[0];rawdiag['logme_half%d'%h]=logs
        costs['S09']=time.monotonic()-clock;clock=time.monotonic()
        gsidx=read(OUT/'WINDOW_MANIFEST.json')[str(s)]['gradient_indices'];gtidx=read(OUT/'WINDOW_MANIFEST.json')[str(t)]['gradient_indices']
        for h in (0,1):
            a=gradient(algo,sd,[i for i in gsidx if hs[i]==h]);b=gradient(algo,td,[i for i in gtidx if ht[i]==h])
            half_results[h]['gradient_cosine']=g.cosine(a,b);del a,b
        torch.cuda.synchronize();costs['S11']=time.monotonic()-clock;clock=time.monotonic()
        geom=read(OLD/'signals'/unit(k,s,t)/'GN_GEOMETRY.json');conf=read(OLD/'SIGNAL_IMPLEMENTATION_PROTOCOL.json')['GN']
        for h in (0,1):
            a,ga=gn_basis(algo,subset(sd,np.where(hs==h)[0].tolist()),geom['weights'],conf)
            b,gb=gn_basis(algo,subset(td,np.where(ht==h)[0].tolist()),geom['weights'],conf)
            r=min(ga['rank'],gb['rank']);half_results[h]['gn_overlap']=float(np.sum((a.T@b)**2)/r) if r else None
            rawdiag['gn_half%d'%h]=dict(source=ga,target=gb);del a,b
        torch.cuda.synchronize();costs['S03']=time.monotonic()-clock
        unchanged=g.compare_states(witness,g.capture(algo));assert all(unchanged.values()),unchanged
        scores={f:mean_optional([q[f] for q in half_results]) for f in half_results[0]}
        # Full v1 GN is retained as a separate reuse diagnostic, not substituted for half aggregation.
        old=read(OLD/'signals'/unit(k,s,t)/'SIGNALS.json')['scores']
        assert abs(avg(affinity,ids(td))-old['A'])<1e-5
        write(dest/'SIGNALS.json',dict(seed=k,source=s,target=t,scores=scores,halves=half_results,details=rawdiag,old_full_gn=old['S_GN'],old_full_affinity=old['A'],repeat_nll_abs_max=float(abs(nll_delta_repeat).max()),state_unchanged=unchanged,seconds=time.monotonic()-begin))
        write(dest/'COMPUTE_COST.json',dict(version=2,package_seconds=costs,scope='incremental after shared B0 forwards; S01/S04/S05/S06 conservative shared cheap bundle; S10 includes cache-read/matching and pre-FiLM extraction amortized over 18 appearances per task in frozen36 rows',matching_cache_amortized_seconds=matching_amortized,model_forward_shared=True,total_seconds=time.monotonic()-begin,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
        npz(STORE/('static_repeat' if repeat_run else 'static')/unit(k,s,t)/'window_values.npz',base_old_nll=bs['nll'],base_target_nll=bt['nll'],source_old_nll=ss['nll'],source_target_nll=st['nll'],expected_base=bt['expected'],expected_source=st['expected'],affinity=affinity,half_source=hs,half_target=ht)
        g.restore(algo,initial);assert all(g.compare_states(initial,g.capture(algo)).values())
        write(dest/'COMPLETE.json',dict(status='PASS',signals_sha256=sha(dest/'SIGNALS.json'),seconds=time.monotonic()-begin,formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

def probe(k,s,t,repeat_run=False):
    import torch
    from task_relation_forecast_v1 import runtime as rt
    from gradient_relation_experiment import kernel as g
    from cb_ifa_experiment import gpu_runtime as env
    dest=OUT/('probes_repeat' if repeat_run else 'probes')/unit(k,s,t);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    sd=rt.load_data(k,s,'D_signal','signal',s,t);td=rt.load_data(k,t,'D_signal','signal',s,t)
    ad=rt.load_data(k,t,'A_train','train',s,t);stream=read(OLD/'sources'/('s%d_t%d'%(k,t))/'STREAM.json')
    algo=rt.build(k,t,STORE/'probe_bootstrap'/unit(k,s,t));begin=time.monotonic();records=[];snapshots={}
    try:
        clean=g.capture(algo)
        for branch in ('source','base'):
            g.restore(algo,clean)
            rt.load_model(algo,rt.model_path(k,'base') if branch=='base' else rt.model_path(k,'source',source=s,step=1000))
            algo.policy.train();initial=g.capture(algo)
            if branch=='source':
                shadow=read(OLD/'signals'/unit(k,s,t)/'SHADOW_ROLLBACK.json')
                assert all(g.digest(initial[f])==shadow['initial_state_witness'][f] for f in ('model','optimizer','scheduler','optimizer_step','scheduler_step','training_modes'))
            baseold=extract(algo,sd)['nll'];basenew=extract(algo,td)['nll'];snapshots[branch]={'0':dict(old=baseold,new=basenew)}
            effective=0;risks=None
            for call in range(50):
                before=tuple(p.detach().clone() for p in algo.policy.parameters() if p.requires_grad)
                algo.policy.train();env.seed_all(stream['policy_seed'][call]);lr=algo.optimizer.param_groups[0]['lr'];assert lr==stream['learning_rates'][call]
                risk_state=g.capture(algo) if branch=='source' and effective==0 and lr>0 else None
                loss=float(g.native(algo,rt.batch(ad,stream['batches'][call]['indices']),t,stream['batches'][call]['epoch']))
                params=tuple(p for p in algo.policy.parameters() if p.requires_grad)
                delta=tuple(p.detach()-v for p,v in zip(params,before));dn=g.norm(delta)
                if lr>0 and dn>0:effective+=1
                history=OLD/('transfers/'+unit(k,s,t) if branch=='source' else 'sources/s%d_t%d'%(k,t))/'steps'/('%04d.json'%(call+1))
                ref=read(history)
                assert ref['record_indices']==stream['batches'][call]['indices'] and ref['lr']==lr
                assert abs(loss-ref['loss'])<=max(1e-6,1e-6*abs(ref['loss'])),('PREFIX_LOSS_MISMATCH',branch,call,loss,ref['loss'])
                assert algo._optimizer_step_count==algo._scheduler_step_count==call+1
                if call==0 and branch=='source':
                    after=g.capture(algo);assert all(g.digest(v)==shadow['after_witness'][f] for f,v in after.items()),'ZERO_STEP_PARITY_MISMATCH'
                if risk_state is not None and effective==1:
                    after=g.capture(algo);g.restore(algo,risk_state);selected=read(OUT/'WINDOW_MANIFEST.json')[str(s)]['gradient_indices'];vv=[]
                    for i in selected:
                        gg=gradient(algo,sd,[i]);vv.append(g.dot(gg,delta));del gg
                    pre_nll=extract(algo,sd)['nll'][selected]
                    risks=dict(values=vv,indices=selected,pre_nll=pre_nll.tolist(),d_norm=dn,normalized_by_base_nll=(np.array(vv)/np.maximum(1,abs(pre_nll))).tolist(),normalization='diagnostic only: h_dot_d / max(1, abs(pre-update window NLL)); raw risk enters registry')
                    g.restore(algo,after);assert all(g.compare_states(after,g.capture(algo)).values());del after,risk_state
                rec=dict(branch=branch,call=call+1,effective_updates=effective,lr=lr,lr_after=algo.optimizer.param_groups[0]['lr'],d_norm=dn,loss=loss,original_loss=ref['loss'],record_indices=ref['record_indices'],optimizer_step=algo._optimizer_step_count,scheduler_step=algo._scheduler_step_count)
                records.append(rec)
                if effective in (1,5,20) and str(effective) not in snapshots[branch]:
                    state=g.capture(algo)
                    snapshots[branch][str(effective)]=dict(old=extract(algo,sd)['nll'],new=extract(algo,td)['nll'],call=call+1,model_sha256=g.digest(state['model']),state_sha256={f:g.digest(v) for f,v in state.items()})
                    assert all(g.compare_states(state,g.capture(algo)).values())
                    save_dir=STORE/('probes_repeat' if repeat_run else 'probes')/unit(k,s,t);save_dir.mkdir(parents=True,exist_ok=True)
                    # Retain the latest valid prefix state, with every q hash/measurement in JSON.
                    env.atomic_save(save_dir/(branch+'_latest_prefix.pt'),dict(state=state,call=call+1,effective_updates=effective,stream_sha256=sha(OLD/'sources'/('s%d_t%d'%(k,t))/'STREAM.json')))
                del before,delta
                if effective>=20:break
            if branch=='source':saved_risks=risks
        results=[{},{}];hs,ht=halves(sd),halves(td)
        for h in (0,1):
            si=hs==h;ti=ht==h
            for q in (1,5,20):
                if str(q) not in snapshots['source'] or str(q) not in snapshots['base']:continue
                a,b=snapshots['source'][str(q)],snapshots['base'][str(q)]
                assert a['call']==b['call'],'EFFECTIVE_COUNTS_DIFFER_REQUIRED_SAME_CALL_EVALUATION'
                old=a['old']-snapshots['source']['0']['old']
                gt=snapshots['source']['0']['new']-a['new'];gb=snapshots['base']['0']['new']-b['new']
                results[h].update({'base_progress%d'%q:avg(gb[ti],ids(td)[ti]),'relative_progress%d'%q:avg((gt-gb)[ti],ids(td)[ti]),'source_damage%d'%q:avg(old[si],ids(sd)[si]),'source_tail%d'%q:tail(old[si],ids(sd)[si])})
            if 'source_damage20' in results[h]:results[h]['damage_5_to_20']=results[h]['source_damage20']-results[h]['source_damage5']
            if saved_risks:
                ix=np.array(saved_risks['indices']);choose=hs[ix]==h;vv=np.array(saved_risks['values'])[choose];ii=ids(sd)[ix][choose]
                tau=32*np.finfo(np.float32).eps*np.maximum(1,abs(np.array(saved_risks['pre_nll'])[choose]))
                # Per-trajectory window upper tail, then upper trajectory tail.
                wt=[]
                for demo in sorted(set(ii)):
                    a=np.sort(vv[ii==demo]);wt.append(float(a[-max(1,math.ceil(.2*len(a))):].mean()))
                wt=np.sort(wt);results[h].update(risk_mean=avg(vv,ii),risk_tail=float(wt[-max(1,math.ceil(.2*len(wt))):].mean()),risk_positive=avg((vv>tau).astype(float),ii))
        scores={f:mean_optional([a.get(f) for a in results]) for f in set().union(*(a.keys() for a in results))}
        serial={br:{q:{f:v.tolist() if isinstance(v,np.ndarray) else v for f,v in a.items()} for q,a in snaps.items()} for br,snaps in snapshots.items()}
        write(dest/'MEASUREMENTS.json',dict(seed=k,source=s,target=t,scores=scores,halves=results,raw_snapshots=serial,risk=saved_risks,seconds=time.monotonic()-begin))
        table(dest/'PROBE_STEPS.csv',records)
        g.restore(algo,clean);assert all(g.compare_states(clean,g.capture(algo)).values())
        write(dest/'COMPLETE.json',dict(status='PASS',measurement_sha256=sha(dest/'MEASUREMENTS.json'),seconds=time.monotonic()-begin,diagnostic_training_performed=True,formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['static','probe'],required=True);p.add_argument('--repeat',action='store_true')
    for f in ('seed','source','target'):p.add_argument('--'+f,type=int,required=True)
    a=p.parse_args()
    import sys
    accessed=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.fsdecode(args[0]).replace('\\','/')
        if 'E_eval' in path or path.endswith(('TARGETS.csv','ROLLOUTS.csv','OOF_PREDICTIONS.csv')) or '/behavior/' in path:
            raise PermissionError('SIGNAL_DATA_FIREWALL: '+path)
        if '/data/' in path or 'reports/' in path or 'diagnostic_storage/' in path:accessed.add(path)
    sys.addaudithook(audit)
    try:(static if a.kind=='static' else probe)(a.seed,a.source,a.target,a.repeat)
    finally:write(OUT/'access'/('%s_%s.json'%(a.kind,unit(a.seed,a.source,a.target))),dict(paths=sorted(accessed),E_eval_accessed=False,outcomes_accessed=False,formal_training_resumed=False))

if __name__=='__main__':main()
