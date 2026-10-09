"""Frozen P1 seed42 -> 200 primary trials -> conditional additional seeds."""
import csv
import fcntl
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from .prepare import ROOT,OUT
from .io import write,sha


def run(cmd,log):
    begin=time.monotonic()
    with open(log,'a') as f:
        p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',pid=os.getpid(),child_pid=p.pid,command=cmd,
              log=str(log),updated_unix=time.time(),e1_enabled=False,baseline_training_rerun=False))
        rc=p.wait()
    with (OUT/'EXECUTION_COST.jsonl').open('a') as f:
        f.write(json.dumps(dict(command=cmd,log=str(log),seconds=time.monotonic()-begin,returncode=rc),allow_nan=False)+'\n')
    if rc:raise RuntimeError('P1 child failed rc=%d log=%s'%(rc,log))


def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


def summarize(seed):
    from .legacy_metrics import fixed_epoch_matrix,metrics
    runroot=ROOT/('runs/p1_object_v1/seed%d'%seed);report=OUT/('seed%d'%seed)
    matrix,_=fixed_epoch_matrix(runroot/'primary_ep99.log')
    rows=[json.loads(s) for s in (report/'trials_primary.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for r in rows:groups[(r['stage'],r['epoch'],r['task'])].append(r)
    assert len(rows)==200 and set(groups)=={(s,99,t) for s in range(6,10) for t in range(6,s+1)}
    reconstructed=[[None]*4 for _ in range(4)]
    for (s,e,t),rs in groups.items():
        assert len(rs)==20 and {r['initial_state_index'] for r in rs}==set(range(20))
        reconstructed[s-6][t-6]=100*sum(r['success'] for r in rs)/20
    assert matrix==reconstructed
    with (report/'training_log.csv').open() as f:epochs=list(csv.DictReader(f))
    assert len(epochs)==400 and {(int(r['stage']),int(r['epoch'])) for r in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
    from .teacher_store import TeacherStore
    cfg=json.loads((OUT/('CONFIG_seed%d.json'%seed)).read_text())
    import torch
    recovery=torch.load(cfg['p1_recovery'],map_location='cpu')
    store=TeacherStore(runroot/'teacher_targets',seed,recovery['teacher_manifest']);store.validate(recovery['buffer_dir'])
    provenance=[]
    prior={}
    for task in range(6,10):
        table=torch.load(runroot/'teacher_targets'/('task%d_ep99.pt'%task),map_location='cpu')
        assert table['seed']==seed and table['source_task']==task and table['epoch']==99
        assert table['checkpoint_sha256']==sha(runroot/'checkpoints'/('mlrifa_%d_model_ep99.pth'%task))
        active=json.loads((runroot/'teacher_targets'/('ACTIVE_task%d.json'%task)).read_text())['windows']
        assert all(v==active[k] for k,v in prior.items() if k in active)
        prior=active
        provenance.append(dict(task=task,windows=len(table['ids']),checkpoint_sha256=table['checkpoint_sha256']))
    values=metrics(matrix);baseline=json.loads((OUT/'PROTOCOL.json').read_text())['baseline_results'][str(seed)];old=metrics(baseline['r'])
    write(report/'metrics_primary.json',dict(r=matrix,**values))
    result=dict(seed=seed,r=matrix,metrics=values,baseline_metrics=old,delta={k:values[k]-old[k] for k in ['auc','fwt','nbt']},
        trials=200,epochs=400,teacher_provenance=provenance,
        training_cost=json.loads((runroot/'TRAINING_COMPLETE.json').read_text()),
        teacher_build_cost=[json.loads(s) for s in (report/'teacher_build_log.jsonl').read_text().splitlines()])
    write(OUT/('RESULT_seed%d.json'%seed),result)
    write(OUT/('FINAL_VERIFICATION_seed%d.json'%seed),dict(status='PASS',trial_matrix_independently_rebuilt=True,
        complete_epochs=400,primary_trials=200,primary_cells=10,immutable_own_seed_teachers=True,
        e1_enabled=False,secondary_evaluation=False,baseline_rerun=False,pretraining_rerun=False))
    if seed==42:write(OUT/'SEED42_DECISION.json',dict(run_additional_seeds=result['delta']['auc']>0,**result))
    write(report/'STATUS.json',dict(status='COMPLETE',seed=seed,p1_training_active=False,primary_trials=200))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert json.loads((OUT/'ENGINEERING_CHECK.json').read_text())['status']=='PASS'
        results=[]
        for seed in [42,43,44]:
            if seed!=42 and not json.loads((OUT/'SEED42_DECISION.json').read_text())['run_additional_seeds']:break
            run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port',str(port()),'-m','p1_object_v1.train','--seed',str(seed)],OUT/('TRAIN_seed%d.log'%seed))
            run(['bash','p1_object_v1/evaluate.sh',str(seed),'primary'],OUT/('EVAL_primary_seed%d.log'%seed))
            results.append(summarize(seed))
        summary=dict(status='COMPLETE',results=results,seeds_completed=len(results),e1_enabled=False,
            baseline_training_rerun=False,primary_gate='POSITIVE_SINGLE_SEED_SIGNAL' if results[0]['delta']['auc']>0 else 'NO_OVERALL_BENEFIT',
            mean={k:statistics.mean(r['metrics'][k] for r in results) for k in ['auc','fwt','nbt']},
            std={k:statistics.stdev(r['metrics'][k] for r in results) if len(results)>1 else None for k in ['auc','fwt','nbt']})
        write(OUT/'SUMMARY.json',summary)
        write(OUT/'FINAL_VERIFICATION.json',dict(status='PASS',seeds=[r['seed'] for r in results],
            conditional_seeds_correct=(len(results)==3)==(results[0]['delta']['auc']>0),primary_only=True))
        lines=['# P1-OBJECT-v1','', 'Fixed beta=0.05, 16 teacher actions/window; original MLR+IFA plus immutable own-stage ep99 GMM distributions. E1 disabled.',
            '', '|Seed|P1 AUC|Historical IFA AUC|Delta AUC|P1 FWT|P1 NBT|','|---|---:|---:|---:|---:|---:|']
        for r in results:lines.append('|%d|%.6f|%.6f|%+.6f|%.6f|%.6f|'%(r['seed'],r['metrics']['auc'],r['baseline_metrics']['auc'],r['delta']['auc'],r['metrics']['fwt'],r['metrics']['nbt']))
        lines+=['','Primary gate: '+summary['primary_gate'], '',
            '200 primary ep99 trials per seed, no secondary checkpoint selection. Baseline, pretraining, and cached features reused.',
            'Teacher targets are immutable own-seed stage-end outputs. Active window count unchanged; teacher numeric payload adds 300 bytes/window plus serialization and index overhead.',
            'Execution costs: EXECUTION_COST.jsonl and RESULT_seed*.json. Distillation forward timing does not include its share of the combined backward; no matched-baseline speed ratio is claimed.',
            'Historical same-seed comparison is not a bitwise paired counterfactual. Seed42 selection is an investment rule, not a significance test. Negative finite-sample KL estimates are retained without clamping.',
            'Stage-level teacher provenance, storage bytes, training diagnostics and trial reconstruction are preserved in per-seed reports.']
        (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
        write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',updated_unix=time.time(),seeds_completed=len(results)))
    except Exception as e:
        write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',error=repr(e),updated_unix=time.time(),pid=os.getpid()))
        raise


if __name__=='__main__':main()
