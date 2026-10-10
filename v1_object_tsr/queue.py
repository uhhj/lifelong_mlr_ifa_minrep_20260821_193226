"""One seed, four native training stages, paired ep99 evaluations, final audit."""
import csv
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from .prepare import ROOT,OUT
from .io import write,sha


def run(cmd,log,phase):
    begin=time.monotonic()
    with open(log,'a') as f:
        child=subprocess.Popen(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',phase=phase,pid=os.getpid(),child_pid=child.pid,
            log=str(log),updated_unix=time.time(),e1_enabled=False,p1_enabled=False,g1_enabled=False))
        rc=child.wait()
    with (OUT/'EXECUTION_COST.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,seconds=time.monotonic()-begin,returncode=rc),allow_nan=False)+'\n')
    if rc:raise RuntimeError('V1 child failed rc=%d log=%s'%(rc,log))


def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


def validate_trials(path,stage,tasks):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
    expected={(stage,99,t,i) for t in tasks for i in range(20)}
    actual=[(r['stage'],r['epoch'],r['task'],r['initial_state_index']) for r in rows]
    assert len(actual)==len(expected) and set(actual)==expected,'INCOMPLETE_OR_DUPLICATE_TRIALS'
    return rows


def evaluate(arm,stage,base_tasks=False):
    view='final_base' if base_tasks else 'primary';name='%s_%s_stage%d'%(arm,view,stage)
    rep=OUT/'seed42'/'evaluation'/name;rep.mkdir(parents=True,exist_ok=True)
    trials=rep/'trials.jsonl';done=rep/'COMPLETE.json';tasks=list(range(6)) if base_tasks else list(range(6,stage+1))
    ckpts=ROOT/('runs/v1_object_tsr/seed42/checkpoints' if arm=='v1' else 'runs/final_paper_text_faithful/ifa/seed42/run_001')
    if done.exists():validate_trials(trials,stage,tasks);return
    if trials.exists() and trials.stat().st_size:
        # Never spend a duplicate trial automatically after partial evaluation.
        raise RuntimeError('PARTIAL_EVALUATION_REQUIRES_EXPLICIT_RECOVERY: '+str(rep))
    os.environ.update(TSR_TRIAL_LOG=str(trials),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',EGL_PLATFORM='surfaceless',MUJOCO_EGL_DEVICE_ID='0',HYDRA_FULL_ERROR='1')
    args=['seed=42','benchmark_name=LIBERO_OBJECT','policy=complete_policy','lifelong=basic',
        'folder='+str(ROOT/'data'),'bddl_folder='+str(ROOT/'libero/libero/bddl_files'),
        'init_states_folder='+str(ROOT/'libero/libero/init_files'),'pretrain='+str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
        'eval.n_eval=20','eval.num_procs=20','+start_task=6','+end_task=10','+aug=true',
        '+epoch_dir='+str(ckpts),'+test_log_file='+str(rep/'success.log'),'training=false',
        '+eval_epochs=[99]','+eval_stage='+str(stage),'+experiment_root='+str(ROOT/'runs/v1_object_tsr/seed42/evaluation'/name)]
    if base_tasks:args+=['+eval_task_ids=[0,1,2,3,4,5]']
    run([sys.executable,'-m','v1_object_tsr.evaluate_spawn_v1']+args,rep/'stdout.log','EVAL_'+name)
    rows=validate_trials(trials,stage,tasks)
    assert all(Path(r['checkpoint'])==ckpts/('mlrifa_%d_model_ep99.pth'%stage) for r in rows)
    # Independent log/individual-trial agreement before marking completion.
    import re
    lines=[tuple(m.groups()) for line in (rep/'success.log').read_text().splitlines() for m in [re.search(r'\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)',line)] if m]
    assert len(lines)==len(tasks)
    for t,e,s,v in lines:assert int(s)==stage and int(e)==99 and abs(float(v)-sum(r['success'] for r in rows if r['task']==int(t))/20)<1e-10
    write(done,dict(status='PASS',arm=arm,stage=stage,tasks=tasks,trials=len(rows),checkpoint_sha256=sha(ckpts/('mlrifa_%d_model_ep99.pth'%stage))))


def summarize():
    from .metrics import exact_metrics,floats
    cfg=json.loads((OUT/'CONFIG_seed42.json').read_text());protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    runroot=Path(cfg['tsr_root']);rep=OUT/'seed42'
    assert all(sha(ROOT/p)==h for p,h in protocol['implementation_files'].items())
    with (rep/'training_log.csv').open() as f:epochs=list(csv.DictReader(f))
    assert len(epochs)==400 and {(int(r['stage']),int(r['epoch'])) for r in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
    import torch
    recovery=torch.load(cfg['tsr_recovery'],map_location='cpu')
    assert recovery['stage_complete'] and recovery['task']==9 and recovery['totals']['steps']==105500
    assert all(int(r['steps'])==protocol['updates_per_epoch'][r['stage']] for r in epochs)
    matrices={};final={};trial_maps={};paired=[]
    for arm in ['v1','baseline']:
        matrix=[[None]*4 for _ in range(4)];tm={}
        for stage in range(6,10):
            rows=validate_trials(rep/'evaluation'/('%s_primary_stage%d'%(arm,stage))/'trials.jsonl',stage,range(6,stage+1))
            for t in range(6,stage+1):matrix[stage-6][t-6]=5*sum(r['success'] for r in rows if r['task']==t)
            tm.update({(r['stage'],r['task'],r['initial_state_index']):r for r in rows})
        rows=validate_trials(rep/'evaluation'/('%s_final_base_stage9'%arm)/'trials.jsonl',9,range(6))
        final[arm]=[5*sum(r['success'] for r in rows if r['task']==t) for t in range(6)]+matrix[-1]
        tm.update({(r['stage'],r['task'],r['initial_state_index']):r for r in rows})
        matrices[arm]=matrix;trial_maps[arm]=tm
    assert trial_maps['v1'].keys()==trial_maps['baseline'].keys()
    for key in sorted(trial_maps['v1']):
        a=trial_maps['v1'][key];b=trial_maps['baseline'][key]
        paired.append(dict(stage=key[0],task=key[1],initial_state=key[2],v1_success=a['success'],baseline_success=b['success']))
    with (OUT/'PAIRED_TRIALS.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(paired[0]));w.writeheader();w.writerows(paired)
    base=exact_metrics(protocol['baseline_results']['42']['r']);v1=exact_metrics(matrices['v1']);paired_base=exact_metrics(matrices['baseline'])
    delta={k:v1[k]-base[k] for k in v1};final_delta=sum(final['v1'])/10-sum(final['baseline'])/10
    screen=delta['auc']>=2 and delta['fwt']>=0 and delta['nbt']<=1 and final_delta>=0
    routes={str(t):json.loads((rep/('ROUTE_task%d.json'%t)).read_text()) for t in range(6,10)}
    signatures={str(t):json.loads((rep/('SIGNATURE_task%d.json'%t)).read_text()) for t in range(6,10)}
    counts={str(t):defaultdict(int) for t in range(6,10)}
    for p in (rep/'sampling').glob('*.json'):
        row=json.loads(p.read_text())
        for t,n in row['actual_counts'].items():counts[str(row['task'])][t]+=n
    write(OUT/'ROUTING_SUMMARY.json',dict(routes=routes,actual_counts={k:dict(v) for k,v in counts.items()},
        signature_norms={k:v['norm'] for k,v in signatures.items()},signature_seconds=sum(v['seconds'] for v in signatures.values()),
        invalid_signatures=[k for k,v in signatures.items() if not v['valid']],
        reference_hashes={v['reference_model_sha256'] for v in signatures.values()} and sorted({v['reference_model_sha256'] for v in signatures.values()})))
    assert len({v['reference_model_sha256'] for v in signatures.values()})==1
    from .signatures import planned_cache
    expected={('t%d__'%t)+p.name for t in range(6,10) for p in planned_cache(cfg['feature_dir'],t,42)}
    assert expected==set(recovery['buffer_files'])=={p.name for p in Path(recovery['buffer_dir']).glob('*.pt')}
    inputs=json.loads((OUT/'INPUT_MANIFEST.json').read_text())
    assert all(sha(p)==h for p,h in inputs['baseline_checkpoints'].items()),'BASELINE_ASSET_CHANGED'
    summary=dict(status='COMPLETE',seed=42,epochs=400,optimizer_updates=105500,rollouts=640,
        matrices=matrices,metrics=floats(v1),historical_metrics=floats(base),paired_baseline_metrics=floats(paired_base),
        historical_delta=floats(delta),paired_delta=floats({k:v1[k]-paired_base[k] for k in v1}),
        final_task_success=final,final_all10_delta=final_delta,
        old_task_drop_review=[t for t in range(9) if final['v1'][t]-final['baseline'][t]<=-10],
        screen='PRIORITY_FOR_MULTISEED_CONFIRMATION' if screen else 'SCREEN_NOT_MET',
        additional_seeds_started=False,baseline_training_rerun=False,
        training_cost=json.loads((runroot/'TRAINING_COMPLETE.json').read_text()))
    write(OUT/'SUMMARY.json',summary)
    write(OUT/'FINAL_VERIFICATION.json',dict(status='PASS',epochs=400,rollouts=640,reference_immutable=True,
        cache_membership_native=True,current_slots_native=True,baseline_checkpoints_readonly=True))
    lines=['# V1-OBJECT-TaskSelectiveReplay','',
        'Seed42 only. Original empty replay and 50% window caches retained by user-approved baseline adaptation; historical source counts 0/1/2/3.',
        'Task6 and task7 retain native sampling; nontrivial source routing starts at task8. BC, IFA, batch slots and 105500 updates unchanged.',
        '', '|Method|AUC|FWT|NBT|Final all-10 SR|','|---|---:|---:|---:|---:|']
    for label,m,fs in [('Historical IFA',floats(base),None),('Paired baseline reevaluation',floats(paired_base),sum(final['baseline'])/10),('V1',floats(v1),sum(final['v1'])/10)]:
        lines.append('|%s|%.9f|%.9f|%.9f|%s|'%(label,m['auc'],m['fwt'],m['nbt'],'not recorded' if fs is None else str(fs)))
    lines+=['','Screen: '+summary['screen'],'Historical comparison uses the unrounded original four-task metric. Final all-10 uses paired evaluation; no baseline training was rerun.',
        'See PAIRED_TRIALS.csv, SUMMARY.json, ROUTING_SUMMARY.json, per-epoch source counts, native IFA diagnostics, and EXECUTION_COST.jsonl.',
        'Routing changes IFA exposure as well as BC exposure. Signed gradient similarity is not proof of positive transfer or shared knowledge.',
        'Single-seed screening is not statistical confirmation. No temperature/coverage search, no additional seeds, no old queues resumed.']
    lines+=['','|Final task|V1|Paired baseline|Delta|','|---|---:|---:|---:|']
    lines+=['|%d|%s|%s|%+g|'%(t,final['v1'][t],final['baseline'][t],final['v1'][t]-final['baseline'][t]) for t in range(10)]
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert json.loads((OUT/'ENGINEERING_CHECK.json').read_text())['status']=='PASS'
        run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port',str(port()),'-m','v1_object_tsr.train','--seed','42'],OUT/'TRAIN_seed42.log','TRAIN_SEED42')
        for arm in ['v1','baseline']:
            for stage in range(6,10):evaluate(arm,stage)
            evaluate(arm,9,True)
        summarize()
        write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',seed=42,updated_unix=time.time(),rollouts=640))
    except Exception as e:
        write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',error=repr(e),updated_unix=time.time(),pid=os.getpid()))
        raise


if __name__=='__main__':main()
