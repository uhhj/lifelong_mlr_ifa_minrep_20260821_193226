"""One authorized E1 sequence, conditional additional E1 seeds; never baseline."""
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .prepare import ROOT,OUT
from .io import write


def run(cmd,log):
    begin=time.monotonic()
    with open(log,'a') as f:
        p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',pid=os.getpid(),child_pid=p.pid,command=cmd,
              log=str(log),updated_unix=time.time(),baseline_training_rerun=False))
        rc=p.wait()
    with (OUT/'EXECUTION_COST.jsonl').open('a') as f:
        f.write(json.dumps(dict(command=cmd,log=str(log),seconds=time.monotonic()-begin,returncode=rc),allow_nan=False)+'\n')
    if rc:raise RuntimeError('E1 child failed rc=%d log=%s'%(rc,log))


def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


def summarize(seed):
    from .legacy_metrics import fixed_epoch_matrix, load_matrix, metrics
    from collections import Counter,defaultdict
    runroot=ROOT/('runs/e1_object_v1/seed%d'%seed)
    matrix,epochs=fixed_epoch_matrix(runroot/'primary_ep99.log')
    values=metrics(matrix)
    baseline=json.loads((OUT/'PROTOCOL.json').read_text())['baseline_results'][str(seed)]
    # Recompute both matrices using the exact existing project's metric routine.
    old=metrics(baseline['r'])
    secondary,selected=load_matrix(runroot/'secondary_selected.log',epochs=(70,75,80,85,90,95,99))
    trial_counts={}
    for view,expected in [('primary',200),('secondary',680)]:
        rows=[json.loads(s) for s in (OUT/('seed%d'%seed)/('trials_%s.jsonl'%view)).read_text().splitlines()]
        groups=defaultdict(list)
        for row in rows:groups[(row['stage'],row['epoch'],row['task'])].append(row)
        assert len(rows)==expected
        assert all(len(rs)==20 and len({r['initial_state_index'] for r in rs})==20 for rs in groups.values())
        if view=='primary':
            assert all(abs(100*sum(r['success'] for r in rs)/20-matrix[s-6][t-6])<1e-9 for (s,e,t),rs in groups.items())
        trial_counts[view]=len(rows)
    stages={}
    for task in range(6,10):
        rows=[]
        for p in sorted((OUT/('seed%d'%seed)/'events').glob('task%d_epoch*.jsonl'%task)):
            if '.attempt' not in p.name:rows.extend(json.loads(s) for s in p.read_text().splitlines())
        n=len(rows);trigger=sum(r['triggered'] for r in rows)
        stages[str(task)]=dict(checks=n,triggered=trigger,accepted=sum(r['accepted'] for r in rows),
            capped=sum(r['capped'] for r in rows),trigger_rate=trigger/n if n else None,
            acceptance_rate=sum(r['accepted'] for r in rows)/trigger if trigger else None,
            cap_rate=sum(r['capped'] for r in rows)/trigger if trigger else None,
            reasons=dict(Counter(r['reason'] for r in rows)),overhead_seconds=sum(r['overhead_seconds'] for r in rows))
    write(OUT/('MECHANISM_seed%d.json'%seed),stages)
    result=dict(seed=seed,r=matrix,metrics=values,baseline_metrics=old,
                delta={k:values[k]-old[k] for k in ('auc','fwt','nbt')},comparison='same seed historical IFA; primary ep99',
                secondary_metrics=metrics(secondary),secondary_selected_epochs=selected,trial_counts=trial_counts,
                mechanism=stages,training_cost=json.loads((runroot/'TRAINING_COMPLETE.json').read_text()))
    write(OUT/('RESULT_seed%d.json'%seed),result)
    if seed==42:write(OUT/'SEED42_DECISION.json',dict(run_additional_seeds=result['delta']['auc']>0,**result))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        check=OUT/'ENGINEERING_CHECK.json'
        if not check.exists() or json.loads(check.read_text())['status']!='PASS':
            run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port',str(port()),'-m','e1_object_v1.smoke'],OUT/'SMOKE.log')
        results=[]
        for seed in [42,43,44]:
            if seed!=42 and not json.loads((OUT/'SEED42_DECISION.json').read_text())['run_additional_seeds']:break
            run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port',str(port()),'-m','e1_object_v1.train','--seed',str(seed)],OUT/('TRAIN_seed%d.log'%seed))
            for view in ['primary','secondary']:
                run(['bash','e1_object_v1/evaluate_spawn_v1.sh',str(seed),view],OUT/('EVAL_%s_seed%d.log'%(view,seed)))
            results.append(summarize(seed))
        import statistics
        summary=dict(status='COMPLETE',results=results,seeds_completed=len(results),baseline_training_rerun=False,
            primary_gate='POSITIVE_SINGLE_SEED_SIGNAL' if results[0]['delta']['auc']>0 else 'NO_OVERALL_BENEFIT',
            mean={k:statistics.mean(x['metrics'][k] for x in results) for k in ('auc','fwt','nbt')},
            std={k:statistics.stdev(x['metrics'][k] for x in results) if len(results)>1 else None for k in ('auc','fwt','nbt')})
        write(OUT/'SUMMARY.json',summary)
        lines=['# E1-OBJECT-v1','', 'Historical MLR+IFA reused; no baseline or pretraining rerun.', '',
            '|Seed|E1 AUC|IFA AUC|Delta AUC|E1 NBT|E1 FWT|','|---|---:|---:|---:|---:|---:|']
        for r in results:lines.append('|%s|%.6f|%.6f|%+.6f|%.6f|%.6f|'%(r['seed'],r['metrics']['auc'],r['baseline_metrics']['auc'],r['delta']['auc'],r['metrics']['nbt'],r['metrics']['fwt']))
        lines+=['','## Mechanism and measured cost','', '|Seed|Checks|Triggers|Accepted|Capped|E1 overhead (s)|Peak GPU (GiB)|','|---|---:|---:|---:|---:|---:|---:|']
        for r in results:
            stages=r['mechanism'].values()
            lines.append('|%d|%d|%d|%d|%d|%.2f|%.3f|'%(r['seed'],sum(s['checks'] for s in stages),sum(s['triggered'] for s in stages),sum(s['accepted'] for s in stages),sum(s['capped'] for s in stages),sum(s['overhead_seconds'] for s in stages),r['training_cost']['peak_gpu_bytes']/2**30))
        lines+=['','Per-stage rates and skip reasons: MECHANISM_seed*.json. Training/evaluation invocation wall times, including interrupted invocations: EXECUTION_COST.jsonl. No baseline speed ratio is asserted without matched timing evidence. Per-trial outcomes and initial-state IDs: seed*/trials_*.jsonl. Secondary selected-checkpoint metrics are in RESULT_seed*.json.']
        lines+=['','Primary gate: '+summary['primary_gate'], '', 'The single-seed gate determines additional E1 investment, not proof of stable improvement. All seeds share a fixed pretrained initialization. Secondary selected-checkpoint evaluation is retained separately and never used for the gate.']
        (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
        write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',updated_unix=time.time(),seeds_completed=len(results)))
    except Exception as e:
        write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',error=repr(e),updated_unix=time.time(),pid=os.getpid()))
        raise


if __name__=='__main__':main()
