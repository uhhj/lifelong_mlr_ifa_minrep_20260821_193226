"""Serial, locked conditional queue: only frozen PRIMARY_EP99 endpoints."""
import csv,fcntl,json,os,re,socket,subprocess,sys,time
from pathlib import Path
from .prepare import ROOT,OUT
from .io import write,sha
from .metrics import exact_metrics,floats


def run(cmd,log,phase):
    begin=time.monotonic()
    with open(log,'a') as f:
        child=subprocess.Popen(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',phase=phase,pid=os.getpid(),child_pid=child.pid,
            log=str(log),updated_unix=time.time(),old_experiment_queues_resumed=False))
        rc=child.wait()
    with (OUT/'EXECUTION_COST.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,seconds=time.monotonic()-begin,returncode=rc),allow_nan=False)+'\n')
    if rc:raise RuntimeError('A1 child failed rc=%d log=%s'%(rc,log))


def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


def validate_trials(path,stage):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
    expected={(stage,99,t,i) for t in range(6,stage+1) for i in range(20)}
    actual=[(r['stage'],r['epoch'],r['task'],r['initial_state_index']) for r in rows]
    assert len(actual)==len(expected) and set(actual)==expected,'INCOMPLETE_OR_DUPLICATE_TRIALS'
    return rows


def evaluate(seed,stage):
    name='primary_stage%d'%stage
    rep=OUT/('seed%d'%seed)/'evaluation'/name;rep.mkdir(parents=True,exist_ok=True)
    trials=rep/'trials.jsonl';done=rep/'COMPLETE.json'
    ckpts=ROOT/('runs/a1_object_v1/seed%d/checkpoints'%seed)
    if done.exists():validate_trials(trials,stage);return
    if trials.exists() and trials.stat().st_size:raise RuntimeError('PARTIAL_EVALUATION_REQUIRES_EXPLICIT_RECOVERY: '+str(rep))
    os.environ.update(A1_TRIAL_LOG=str(trials),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',EGL_PLATFORM='surfaceless',MUJOCO_EGL_DEVICE_ID='0',HYDRA_FULL_ERROR='1')
    args=['seed=%d'%seed,'benchmark_name=LIBERO_OBJECT','policy=complete_policy','lifelong=basic',
        'folder='+str(ROOT/'data'),'bddl_folder='+str(ROOT/'libero/libero/bddl_files'),
        'init_states_folder='+str(ROOT/'libero/libero/init_files'),'pretrain='+str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
        'eval.n_eval=20','eval.num_procs=20','+start_task=6','+end_task=10','+aug=true',
        '+epoch_dir='+str(ckpts),'+test_log_file='+str(rep/'success.log'),'training=false',
        '+eval_epochs=[99]','+eval_stage='+str(stage),'+experiment_root='+str(ROOT/('runs/a1_object_v1/seed%d/evaluation'%seed)/name)]
    run([sys.executable,'-m','a1_object_v1.evaluate_spawn_v1']+args,rep/'stdout.log','EVAL_SEED%d_STAGE%d'%(seed,stage))
    rows=validate_trials(trials,stage)
    assert all(Path(r['checkpoint'])==ckpts/('mlrifa_%d_model_ep99.pth'%stage) for r in rows)
    lines=[tuple(m.groups()) for line in (rep/'success.log').read_text().splitlines() for m in [re.search(r'\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)',line)] if m]
    assert len(lines)==stage-5
    for t,e,s,v in lines:assert int(s)==stage and int(e)==99 and abs(float(v)-sum(r['success'] for r in rows if r['task']==int(t))/20)<1e-10
    write(done,dict(status='PASS',seed=seed,stage=stage,trials=len(rows),checkpoint_sha256=sha(ckpts/('mlrifa_%d_model_ep99.pth'%stage))))


def summarize(seed):
    import torch
    cfg=json.loads((OUT/('CONFIG_seed%d.json'%seed)).read_text());protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in protocol['implementation_files'].items())
    rep=OUT/('seed%d'%seed)
    epochs=list(csv.DictReader((rep/'training_log.csv').open()))
    assert len(epochs)==400 and {(int(r['stage']),int(r['epoch'])) for r in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
    recovery=torch.load(cfg['a1_recovery'],map_location='cpu')
    assert recovery['stage_complete'] and recovery['task']==9
    assert sum(int(r['steps']) for r in epochs)==recovery['totals']['steps']
    for p in (rep/'sampling').glob('*.json'):
        s=json.loads(p.read_text());assert s['direct_native'] and s['original_index_sha256']==s['routed_index_sha256']
    matrix=[[None]*4 for _ in range(4)]
    for stage in range(6,10):
        rows=validate_trials(rep/'evaluation'/('primary_stage%d'%stage)/'trials.jsonl',stage)
        for t in range(6,stage+1):matrix[stage-6][t-6]=5*sum(r['success'] for r in rows if r['task']==t)
    base=json.loads((OUT/'BASELINE_REFERENCE.json').read_text())['results'][str(seed)]['matrix']
    m,b=exact_metrics(matrix),exact_metrics(base)
    # Independently recompute all three metrics from the reconstructed matrix.
    alt_auc=sum(sum(matrix[i][j] for i in range(j,4))/(4-j) for j in range(4))/4
    alt_fwt=sum(matrix[j][j] for j in range(4))/4
    alt_nbt=sum(sum(matrix[j][j]-matrix[i][j] for i in range(j+1,4))/(3-j) for j in range(3))/3
    assert max(abs(float(m[k])-v) for k,v in [('auc',alt_auc),('fwt',alt_fwt),('nbt',alt_nbt)])<1e-10
    result=dict(status='COMPLETE',seed=seed,matrix=matrix,baseline_matrix=base,metrics=floats(m),baseline_metrics=floats(b),
        delta=floats({k:m[k]-b[k] for k in m}),exact_delta={k:str(m[k]-b[k]) for k in m},
        positive_auc=bool(m['auc']>b['auc']),epochs=400,primary_trials=200,optimizer_updates=recovery['totals']['steps'])
    write(OUT/('RESULT_seed%d.json'%seed),result)
    mechanism={}
    for stage in range(6,10):
        n,small,zn,dn,rho,energy,ms,pairs,active=0,0,0.,0.,0.,0.,0.,0,0.
        native_cmp=[];batches=0;clip=0
        for p in sorted((rep/'events').glob('task%d_epoch*.jsonl'%stage)):
            for line in p.open():
                row=json.loads(line);d=row.get('a1',{});batches+=1;clip+=row['gradient_norm']>cfg['train']['grad_clip']
                if 'count' not in d:continue
                n+=d['count'];small+=d['small_projection_count'];zn+=d['z_norm_sum'];dn+=d['projected_norm_sum']
                rho+=d['rho_sum'];energy+=d['energy_ratio_sum'];ms+=d['projection_ms']
                for v in d.get('pairs',[]):
                    pairs+=1;active+=v['active_fraction']
                    if 'same_z_global_hinge_mean' in v:native_cmp.append(dict(step=row['step'],global_hinge=v['same_z_global_hinge_mean'],projected_hinge=v['hinge_mean']))
        sub=json.loads((rep/('SUBSPACE_stage%d.json'%stage)).read_text())
        assert sub['task_ids']==list(range(stage+1)) and sub['rank']<=stage+1
        mechanism[stage]=dict(subspace=sub,projection_window_count=n,small_projection_fraction=small/n if n else None,
            mean_z_norm=zn/n if n else None,mean_projected_norm=dn/n if n else None,mean_rho=rho/n if n else None,
            mean_energy_fraction=energy/n if n else None,projection_seconds=ms/1000,
            ifa_pair_batch_count=pairs,mean_active_fraction=active/pairs if pairs else None,
            clip_fraction=clip/batches,same_z_diagnostics=native_cmp)
    write(OUT/('MECHANISM_seed%d.json'%seed),mechanism)
    write(OUT/('VERIFICATION_seed%d.json'%seed),dict(status='PASS',epochs=400,trials=200,matrix_cells=10,
        native_sampling=True,independent_metrics=True,only_arrived_references=True,own_reference_fix=True))
    return result


def final_report(results,status):
    import statistics
    rows=['# A1-OBJECT-v1','', 'Status: '+status,
        'Corrected own-reference IFA baseline. Only raw IFA features are projected; actions use the original features.',
        'Historical baseline was trained on RTX4090; this run uses the recorded current environment. No baseline retraining or outcome-matching reruns.',
        '', '|Seed|A1 AUC|Corrected IFA AUC|Delta|FWT|NBT|','|---|---:|---:|---:|---:|---:|']
    for r in results:rows.append('|%d|%.9f|%.9f|%+.9f|%.9f|%.9f|'%(r['seed'],r['metrics']['auc'],r['baseline_metrics']['auc'],r['delta']['auc'],r['metrics']['fwt'],r['metrics']['nbt']))
    rows += ['', 'A positive single-seed delta is a screening signal, not evidence of significance or shared-skill decomposition.',
        'No E1/P1/G1/R1/A2/A3, no replay reweighting, no baseline/pretraining rerun. No auxiliary evaluation.',
        'See per-seed raw rollout records, RESULT, MECHANISM and VERIFICATION files.']
    if len(results)==3:
        rows+=['','Three-seed AUC mean %.9f; sample std %.9f; mean paired delta %+.9f.'%(
            statistics.mean(r['metrics']['auc'] for r in results),statistics.stdev(r['metrics']['auc'] for r in results),statistics.mean(r['delta']['auc'] for r in results))]
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf8')
    write(OUT/'SUMMARY.json',dict(status=status,results=results,baseline_training_rerun=False,old_experiment_queues_resumed=False))
    costs=[json.loads(x) for x in (OUT/'EXECUTION_COST.jsonl').read_text().splitlines()]
    write(OUT/'EXECUTION_COST.json',dict(invocations=costs,training={str(r['seed']):json.loads((ROOT/('runs/a1_object_v1/seed%d/TRAINING_COMPLETE.json'%r['seed'])).read_text()) for r in results},
        reference_generation=json.loads((OUT/'REFERENCE_MANIFEST.json').read_text())['seconds'],
        cost_includes_device_pre_update_snapshot=True,baseline_matching_timing_available=False))
    write(OUT/'FINAL_VERIFICATION.json',dict(status='PASS',completed_seeds=[r['seed'] for r in results],primary_trials=200*len(results),
        conditional_seed_rule_respected=len(results)==1 or results[0]['positive_auc'],
        scope='OBJECT completed seeds; cross-benchmark expansion is separately prerequisite-gated'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert json.loads((OUT/'ENGINEERING_CHECK.json').read_text())['status']=='PASS'
        results=[]
        for seed in (42,43,44):
            if seed>42 and not results[0]['positive_auc']:break
            run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port',str(port()),'-m','a1_object_v1.train','--seed',str(seed)],OUT/('TRAIN_seed%d.log'%seed),'TRAIN_SEED%d'%seed)
            for stage in range(6,10):evaluate(seed,stage)
            results.append(summarize(seed))
            final_report(results,'SEED42_SCREEN_COMPLETE' if seed==42 else 'OBJECT_CONFIRMATION_RUNNING')
        mean_delta=sum(r['delta']['auc'] for r in results)/len(results)
        status='OBJECT_COMPLETE_SCREEN_NOT_MET' if mean_delta<=0 else 'OBJECT_COMPLETE_EXPANSION_PREREQUISITES_REQUIRED'
        final_report(results,status)
        # No GOAL/LIBERO50 reference-consistent baseline is inferred from OBJECT.
        # Persist the exact requirement instead of running an unmatched comparator.
        if len(results)==3 and mean_delta>0:
            write(OUT/'EXPANSION_REQUIRED.json',dict(status='PENDING_VERIFIED_INPUTS',datasets=['LIBERO_GOAL','LIBERO_50'],
                required='Same-reference corrected baseline matrices/configs, pretrained language references and canonical grouped task loader; no baseline rerun authorized.'))
        write(OUT/'QUEUE_STATUS.json',dict(status=status,completed_seeds=[r['seed'] for r in results],updated_unix=time.time(),primary_trials=200*len(results)))
    except Exception as e:
        write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',error=repr(e),updated_unix=time.time(),pid=os.getpid()))
        raise


if __name__=='__main__':main()
