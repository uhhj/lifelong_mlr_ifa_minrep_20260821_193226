"""Independent raw-event/trial integrity checks before descriptive single-seed report."""
import csv
import json
import statistics
from .common import *

def main():
    from fixed_kappa_experiment.metrics import read_fixed_matrix,score_matrix
    verify();training=read(RUN/'TRAINING_COMPLETE.json');assert training['status']=='PASS'
    assert read(RUN/'EVALUATION_COMPLETE.json')['status']=='PASS'
    epochs=read(RUN/'ACCEPTED_EPOCHS.json');assert len(epochs)==400
    assert {(r['task'],r['epoch']) for r in epochs}=={(t,e) for t in TASKS for e in range(100)}
    expected_global=0;diagnostics=[]
    for row in epochs:
        assert sha(row['event_path'])==row['event_sha256']
        events=[json.loads(s) for s in Path(row['event_path']).read_text().splitlines()]
        assert len(events)==row['steps']
        for j,e in enumerate(events):
            expected_global+=1
            assert e['global_step']==expected_global and e['task']==row['task'] and e['epoch']==row['epoch']
            assert e['batch']==j and e['optimizer_step']==row['epoch']*row['steps']+j+1
            assert e['constraints']==32*(row['task']-6)
        diagnostics.append(dict(task=row['task'],epoch=row['epoch'],steps=len(events),
            trigger_count=sum(e['trigger'] for e in events),
            mean_correction_ratio=statistics.mean(e.get('correction_ratio',0) for e in events),
            seconds=row['seconds'],mean_bc=statistics.mean(e['mixed_bc'] for e in events)))
    assert expected_global==105500
    r=[[None]*4 for _ in range(4)];trial_rows=[]
    for stage in TASKS:
        for task in range(6,stage+1):
            x=read(RUN/'evaluation'/('stage%d_task%d.json'%(stage,task)))
            assert len(x['trials'])==20 and [t['trial'] for t in x['trials']]==list(range(20))
            successes=sum(t['success'] for t in x['trials'])
            assert successes/20==x['success_rate'];r[stage-6][task-6]=successes*5.
            trial_rows.extend([dict(seed=46,stage=stage,task=task,**t) for t in x['trials']])
    assert r==read_fixed_matrix(RUN/'primary_ep99.log')
    scores=score_matrix(r)
    # Recompute primary metrics independently of the project's scoring helper.
    fwt=sum(r[i][i] for i in range(4))/4
    auc=sum(sum(r[s][t] for s in range(t,4))/(4-t) for t in range(4))/4
    nbt=sum(sum(r[t][t]-r[s][t] for s in range(t+1,4))/(3-t) for t in range(3))/3
    assert max(abs(scores[k]-v) for k,v in [('fwt',fwt),('auc',auc),('nbt',nbt)])<1e-10
    per_task=[dict(task=t+6,first=r[t][t],final=r[-1][t],
        column_auc=sum(r[s][t] for s in range(t,4))/(4-t),
        peak_to_final=max(r[s][t] for s in range(t,4))-r[-1][t]) for t in range(4)]
    write(OUT/'RESULTS.json',dict(seed=46,method='SAMPLE_STEP_QP',metrics=scores,per_task=per_task))
    for name,rows in [('trials.csv',trial_rows),('epochs.csv',diagnostics),('per_task.csv',per_task)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    historical=ROOT/'runs/fixed_kappa_confirmation_v2/formal/seed46/mlr/primary_ep99.log'
    history=score_matrix(read_fixed_matrix(historical)) if historical.exists() else None
    write(OUT/'HISTORICAL_CONTEXT.json',dict(metrics=history,source=str(historical),
          interpretation='Unpaired cross-environment developmental context; not a controlled superiority comparison'))
    write(OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS',epochs=400,updates=expected_global,
          cells=10,trials=200,raw_trial_matrix_equals_native_log=True,independent_metrics_exact=True))
    write(OUT/'DECISION.json',dict(decision='SINGLE_SEED_PILOT_COMPLETE',
        method_superiority='NOT_ESTABLISHED',reason='One development seed, QP-only, no same-environment MLR control'))
    lines=['# Full lifelong per-sample QP pilot','',
        'Completed seed46 only. Fixed ep99; 400 epochs, 105500 updates and 200 native closed-loop trials.','',
        '| Method | FWT | NBT | AUC | Final success |','|---|---:|---:|---:|---:|',
        '| SAMPLE_STEP_QP | %.2f | %.2f | %.2f | %.2f |'%(scores['fwt'],scores['nbt'],scores['auc'],scores['final_mean_success']),
        '', '| Task | First | Final | Column AUC | Peak-to-final |','|---|---:|---:|---:|---:|']
    lines += ['| %(task)d | %(first).2f | %(final).2f | %(column_auc).2f | %(peak_to_final).2f |'%x for x in per_task]
    lines += ['', 'Execution integrity: raw events, all 10 cells, all 200 trial flags, native evaluation log and independently recomputed metrics agree.',
        '', 'Historical MLR seed46 is separate unpaired context: '+json.dumps(history),
        '', 'Limitations: one reused development seed; fixed stage references may overfit retained windows; historical results use another environment. '
        'No method superiority, training-seed uncertainty or statistical significance claim. Each cell has only 20 trials (5 percentage-point resolution). '
        'The project calls mean first-learning diagonal success FWT; it is not a zero-shot transfer measurement.',
        '', 'Decision: SINGLE_SEED_PILOT_COMPLETE. A same-environment MLR control is required before attributing differences to the method.']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'STATUS.json',dict(status='COMPLETE',decision='SINGLE_SEED_PILOT_COMPLETE',time=time.time()))

if __name__=='__main__':main()
