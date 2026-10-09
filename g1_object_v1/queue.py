"""Frozen G1 seed42 -> 200 primary trials -> conditional additional seeds."""
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
    if rc:raise RuntimeError('G1 child failed rc=%d log=%s'%(rc,log))


def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


def summarize(seed):
    from .legacy_metrics import fixed_epoch_matrix,metrics
    runroot=ROOT/('runs/g1_object_v1/seed%d'%seed);report=OUT/('seed%d'%seed)
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
    import torch
    cfg=json.loads((OUT/('CONFIG_seed%d.json'%seed)).read_text())
    recovery=torch.load(cfg['g1_recovery'],map_location='cpu')
    assert recovery['stage_complete'] and recovery['task']==9
    assert recovery['buffer_files']==sorted(p.name for p in Path(recovery['buffer_dir']).glob('*.pt'))
    assert all(sha(ROOT/p)==h for p,h in json.loads((OUT/'PROTOCOL.json').read_text())['implementation_files'].items())
    assert all(Path(row['checkpoint'])==runroot/'checkpoints'/('mlrifa_%d_model_ep99.pth'%row['stage']) for row in rows)
    mechanism={}
    for task in range(6,10):
        records=[]
        for epoch in range(100):
            records.extend(json.loads(line) for line in (report/'events'/('task%d_epoch%03d.jsonl'%(task,epoch))).read_text().splitlines())
        nonzero=sum(r['has_ifa'] and not r['hinge_zero'] for r in records)
        decisions=sum(r['dot'] is not None for r in records)
        closed=sum(r['w']==0 for r in records)
        proposed=sum(sum(r['weighted_terms']) for r in records);retained=sum(r['retained_ifa'] for r in records)
        mechanism[str(task)]=dict(steps=len(records),ifa_constructed=sum(r['has_ifa'] for r in records),
            nonzero_ifa=nonzero,direction_decisions=decisions,actively_closed=closed,
            zero_gradients=sum(r['reason']=='ZERO_GRADIENT' for r in records),
            closure_rate=closed/decisions if decisions else None,closure_denominator='direction_decisions',
            proposed_ifa=proposed,retained_ifa=retained,R_IFA=retained/proposed if proposed else None,
            probe_seconds=sum(r['probe_seconds'] for r in records),dot_seconds=sum(r['dot_seconds'] for r in records),
            peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in records))
        assert len(records)==sum(int(e['steps']) for e in epochs if int(e['stage'])==task)
        assert [r['step'] for r in records]==list(range(1,len(records)+1))
    assert mechanism['6']['direction_decisions']==0
    write(OUT/('MECHANISM_seed%d.json'%seed),dict(stages=mechanism,totals=recovery['totals'],
        interpretation='Own G1 trajectory scalar retention, not a baseline counterfactual'))
    from .summary import build_result
    baseline=json.loads((OUT/'PROTOCOL.json').read_text())['baseline_results'][str(seed)]
    result=build_result(seed,matrix,baseline['r'])
    result.update(trials=200,epochs=400,mechanism=mechanism,
        training_cost=json.loads((runroot/'TRAINING_COMPLETE.json').read_text()))
    write(report/'metrics_primary.json',dict(r=matrix,**result['metrics']))
    write(OUT/('RESULT_seed%d.json'%seed),result)
    write(OUT/('FINAL_VERIFICATION_seed%d.json'%seed),dict(status='PASS',trial_matrix_independently_rebuilt=True,
        complete_epochs=400,primary_trials=200,primary_cells=10,native_buffer_only=True,
        e1_enabled=False,secondary_evaluation=False,baseline_rerun=False,pretraining_rerun=False))
    if seed==42:write(OUT/'SEED42_DECISION.json',result)
    write(report/'STATUS.json',dict(status='COMPLETE',seed=seed,g1_training_active=False,primary_trials=200))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert json.loads((OUT/'ENGINEERING_CHECK.json').read_text())['status']=='PASS'
        results=[]
        for seed in [42,43,44]:
            if seed!=42 and not json.loads((OUT/'SEED42_DECISION.json').read_text())['run_additional_seeds']:break
            run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port',str(port()),'-m','g1_object_v1.train','--seed',str(seed)],OUT/('TRAIN_seed%d.log'%seed))
            run(['bash','g1_object_v1/evaluate.sh',str(seed),'primary'],OUT/('EVAL_primary_seed%d.log'%seed))
            results.append(summarize(seed))
        summary=dict(status='COMPLETE',results=results,seeds_completed=len(results),e1_enabled=False,
            baseline_training_rerun=False,primary_gate='POSITIVE_SINGLE_SEED_SIGNAL' if results[0]['run_additional_seeds'] else 'NO_OVERALL_BENEFIT',
            mean={k:statistics.mean(r['metrics'][k] for r in results) for k in ['auc','fwt','nbt']},
            std={k:statistics.stdev(r['metrics'][k] for r in results) if len(results)>1 else None for k in ['auc','fwt','nbt']})
        write(OUT/'SUMMARY.json',summary)
        write(OUT/'FINAL_VERIFICATION.json',dict(status='PASS',seeds=[r['seed'] for r in results],
            conditional_seeds_correct=(len(results)==3)==(results[0]['run_additional_seeds']),primary_only=True))
        from .summary import render
        (OUT/'FINAL_REPORT.md').write_text(render(results),encoding='utf8')
        write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',updated_unix=time.time(),seeds_completed=len(results)))
    except Exception as e:
        write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',error=repr(e),updated_unix=time.time(),pid=os.getpid()))
        raise


if __name__=='__main__':main()
