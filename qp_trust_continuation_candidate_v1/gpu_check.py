import contextlib,gc,io,json,os,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import step,core,solver
from .io import write_json
from qp_lbfgsb_kkt_driven_v1.offline import OUT,save_npz
from sample_qp_lifelong_v1_continuation_v3 import common as c
from sample_qp_lifelong_v1_continuation_v3.step import update as oldupdate

STORE=Path('/root/qp_trust_continuation_candidate_v1')

def save(path,value):
    assert not path.exists(),path
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def run_hard(g,algo,ds,refs,name,path):
    saved=torch.load(path,map_location='cpu');base=saved['state'];indices=saved['record_indices']
    manifest_key='reference_sha' if name=='B258' else 'reference_sha256'
    assert c.sha(c.OUT/'plans/references_task8.json')==saved[manifest_key]
    data=g.batch(ds,indices)
    if name=='B185':assert [int(v) for v in data[1]]==saved['source_ids']
    else:
        from qp_v3_failure_capture_v1.run import fingerprint
        expected=c.read(c.ROOT/'reports/qp_v3_failure_capture_v1/t8_e11_b258/UPDATE_INPUT_FINGERPRINT.json')
        assert fingerprint(data,'batch')==expected['batch']
        rp=[]
        for ri,rb in enumerate(refs):
            for wi in range(len(rb['feature'])):rp.extend(fingerprint({k:v[wi] for k,v in rb.items() if torch.is_tensor(v)},'ref/%d/%d'%(ri,wi)))
        assert rp==expected['reference_windows']
    g.restore(algo,base);restored=g.compare_states(base,g.capture(algo));assert all(restored.values())
    before_opt=algo._optimizer_step_count;before_sched=algo._scheduler_step_count
    native_states=[];projected=[];native=g.native;project=core.project
    def native_once(*args,**kw):
        result=native(*args,**kw);native_states.append(g.capture(algo));return result
    def checked_project(H,K,displacement,*args,**kw):
        correction,sol=project(H,K,displacement,*args,**kw)
        projected.append(dict(solution=sol,correction_finite=bool(torch.isfinite(correction).all()),displacement_finite=bool(torch.isfinite(displacement).all()),gram=K.copy()))
        return correction,sol
    w=core.Workspace(g.params(algo));folder=OUT/'writeback'/name;folder.mkdir(parents=True,exist_ok=True)
    with patch.object(g,'native',side_effect=native_once),patch.object(step,'project',side_effect=checked_project):
        rec=step.update(algo,data,8,11,refs,w,diag_dir=folder/'qp',context=dict(batch=saved['batch'],pre_global_step=saved['global_step'],record_indices=indices,source_ids=list(data[1]),reference_sha256=saved[manifest_key]))
    final=g.capture(algo);sol=projected[0]['solution'];native_state=native_states[0]
    states=g.compare_states(native_state,final)
    checks=dict(restored=all(restored.values()),native_calls_once=len(native_states)==1,project_calls_once=len(projected)==1,optimizer_counter_once=algo._optimizer_step_count==before_opt+1,scheduler_counter_once=algo._scheduler_step_count==before_sched+1,optimizer_moments_unchanged_by_projection=states['optimizer'],scheduler_unchanged_by_projection=states['scheduler'],rng_unchanged_by_projection=states['rng'],modes_unchanged=states['training_modes'],physical_gate=sol['written_normalized_rate_max']<=5e-6,lambda_nonnegative=bool((np.asarray(sol['lam'])>=0).all()),parameter_finite=all(torch.isfinite(p).all().item() for p in algo.policy.parameters()),correction_finite=projected[0]['correction_finite'],displacement_finite=projected[0]['displacement_finite'],actual_positive_zero=rec['actual_positive']==0,expected_trust=bool(sol['second_fallback']['triggered'])==(name=='B258'),kkt_pass=sol['kkt_error']<=1e-8)
    expected_standard=1.0093558544532755e-8 if name=='B258' else 4.864608038612772e-8
    checks['standard_exact']=sol['standard_kkt']==expected_standard
    checks['lbfgsb_kkt_exact']=sol['fallback']['fallback_kkt']==(1.547045713756745e-8 if name=='B258' else 4.1823128293927425e-10)
    # Every optimizer tensor step, not just the bookkeeping counter.
    steps=[]
    for key,val in base['optimizer']['state'].items():
        if 'step' in val:
            old=float(val['step']);new=float(final['optimizer']['state'][key]['step']);steps.append(new-old)
    checks['every_adamw_state_step_once']=bool(steps) and all(v==1 for v in steps)
    if name=='B185':
        historical=torch.load(c.OUT/'regression/b185_writeback/post_step_state.pt',map_location='cpu')['state'];parity=g.compare_states(historical,final);checks['historical_v3_final_state_exact']=all(parity.values());del historical
    else:parity=None
    post=STORE/(name+'_POST_STEP_STATE.pt');save(post,dict(state=final,task=8,epoch=11,batch=saved['batch'],global_step=saved['global_step']+1,record_indices=indices,source='diagnostic single-step; not canonical epoch checkpoint'))
    loaded=torch.load(post,map_location='cpu');g.restore(algo,loaded['state']);reload=g.compare_states(final,g.capture(algo));checks['post_state_restore']=all(reload.values())
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,record=rec,solver=sol,restore=restored,reload=reload,history_parity=parity,optimizer_before=before_opt,optimizer_after=algo._optimizer_step_count,scheduler_before=before_sched,scheduler_after=algo._scheduler_step_count,optimizer_step_deltas=steps,post_state=str(post),post_state_sha256=c.sha(post),formal_training_resumed=False)
    write_json(folder/'WRITEBACK.json',report);assert all(checks.values()),checks
    print('WRITEBACK_PASS',name,rec['kkt'],rec['projection_residual'],flush=True)
    del final,native_states,native_state,loaded,base,saved,w,projected;gc.collect();torch.cuda.empty_cache()
    return report

def main():
    from functional_geometry_diagnostic_v1 import geometry as g
    from cb_ifa_experiment.gpu_runtime import bootstrap_train,build_train
    assert c.read(OUT/'HARD_QP_REGRESSION_V3.json')['status']=='PASS'
    assert core.solve is solver.solve and step.project is core.project
    STORE.mkdir(exist_ok=True);g.init_ddp(STORE/'ddp');torch.set_num_threads(2);torch.set_num_interop_threads(1)
    try:
        cfg=c.config(STORE/'bootstrap');bootstrap_train(cfg);algo=build_train(cfg);initial=g.cpu_state(algo.policy.state_dict())
        plan,ds,sampler,loader=c.dataset_loader(8)
        with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,loader)
        refs=c.reference_batches(algo,8)
        results={}
        for name,path in [('B258',Path('/root/qp_v3_failure_capture_v1/PRE_FAILURE_STATE.pt')),('B185',c.ROOT/'reports/qp_later_failure_diag_v1/t8_e11_b180/PRE_FAILURE_STATE.pt')]:
            write_json(OUT/'STATUS.json',dict(stage='GPU_WRITEBACK',fixture=name,formal_training_resumed=False));results[name]=run_hard(g,algo,ds,refs,name,path)
        del ds,loader,refs;gc.collect();torch.cuda.empty_cache()
        normal=[]
        for task in (7,8,9):
            write_json(OUT/'STATUS.json',dict(stage='GPU_NORMAL_PARITY',task=task,formal_training_resumed=False))
            plan,ds,sampler,loader=c.dataset_loader(task);algo.policy.load_state_dict(initial)
            with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,loader)
            algo.policy.train();refs=c.reference_batches(algo,task);order=list(iter(sampler));data=g.batch(ds,order[:10]);base=g.capture(algo)
            records={};states={}
            for arm,fn in [('v3',oldupdate),('candidate',step.update)]:
                g.restore(algo,base);w=core.Workspace(g.params(algo));records[arm]=fn(algo,data,task,0,refs,w);states[arm]=g.capture(algo);del w
            cmp=g.compare_states(states['v3'],states['candidate']);r=records['candidate']
            checks=dict(state_exact=all(cmp.values()),standard_pass=r['kkt']<=1e-8,lbfgsb_not_called=not r['fallback_triggered'],trust_not_called=not r['second_fallback_triggered'])
            normal.append(dict(task=task,constraints=32*(task-6),checks=checks,state_checks=cmp,records=records,status='PASS' if all(checks.values()) else 'FAIL'));write_json(OUT/'NORMAL_GPU_PARITY.json',dict(results=normal,status='RUNNING',formal_training_resumed=False));assert all(checks.values()),normal[-1]
            print('NORMAL_PARITY_PASS',task,flush=True)
            del ds,loader,refs,base,states,data;gc.collect();torch.cuda.empty_cache()
        write_json(OUT/'NORMAL_GPU_PARITY.json',dict(results=normal,status='PASS',formal_training_resumed=False))
        write_json(OUT/'GPU_CHECK.json',dict(status='PASS',hard_writebacks={k:v['status'] for k,v in results.items()},normal_counts=[32,64,96],formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(OUT/'GPU_ERROR.json',dict(traceback=traceback.format_exc(),formal_training_resumed=False));raise
