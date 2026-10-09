"""Read-only scientific checks and historical E0 comparison; no policy imports."""
import csv,json,os,time
from pathlib import Path
from collections import Counter,defaultdict
from .io import write,sha
from .legacy_metrics import metrics,fixed_epoch_matrix,load_matrix

S=Path(os.environ['E1_SOURCE']);O=Path(os.environ['E1_OUT'])
def auc(r):return sum(sum(r[q][i] for q in range(i,4))/(4-i) for i in range(4))/4
def main():
    assert json.loads((O/'TRAINING_COMPLETE.json').read_text())['status']=='TRAINED_EVAL_PENDING'
    for v,n in [('primary',200),('secondary',680)]:
        receipt=json.loads((O/(v+'_COMPLETE.json')).read_text())
        path=O/('trials_'+v+'.jsonl');assert sha(path)==receipt['trials_sha256']
        rows=[json.loads(x) for x in path.read_text().splitlines()];assert len(rows)==n
        groups=defaultdict(list)
        for x in rows:groups[x['stage'],x['epoch'],x['task']].append(x)
        assert all(len(xs)==20 and sorted(r['initial_state_index'] for r in xs)==list(range(20)) for xs in groups.values())
        # Secondary may reuse an already-evaluated ep99 cell in a different view; views stay separate.
        assert len(groups)==(10 if v=='primary' else 34)
        matrix,selected=fixed_epoch_matrix(O/'eval_primary.log') if v=='primary' else load_matrix(O/'eval_secondary.log',epochs=(70,75,80,85,90,95,99))
        for stage in range(6,10):
            for task in range(6,stage+1):
                xs=groups[stage,int(selected[str(stage)]),task]
                assert abs(sum(x['success'] for x in xs)*5-matrix[stage-6][task-6])<1e-8
        write(O/('metrics_'+v+'.json'),dict(metrics(matrix),selected_epochs=selected))
    rows=list(csv.DictReader((O/'run_001/paper_repro_train_curve.csv').open()))
    assert len(rows)==400 and len({(r['stage'],r['epoch']) for r in rows})==400
    assert all(int(r['batch_size'])==10 for r in rows)
    expected_batches={6:157,7:227,8:309,9:362}
    for r in rows:assert int(r['batches'])==expected_batches[int(r['stage'])]
    contract=json.loads((O/'CONTRACT.json').read_text())
    assert all(sha(S/k)==v for k,v in contract['source_sha256'].items())
    parity=json.loads((O/'FEATURE_PARITY.json').read_text())
    for t,files in parity['features'].items():
        assert all(sha(O/'features'/t/n)==h for n,h in files.items())
    training=json.loads((O/'TRAINING_COMPLETE.json').read_text())
    assert all(sha(O/'run_001'/n)==h for n,h in training['checkpoints'].items())
    baseline=json.loads((S/'protective_e1/e0_seed42.json').read_text());new=json.loads((O/'metrics_primary.json').read_text())
    delta={k:new[k]-baseline[k] for k in ('auc','fwt','nbt')}
    improved=auc(new['r'])>auc(baseline['r'])+1e-9
    result=dict(status='SEED42_COMPLETE_EXTENSION_PENDING' if improved else 'COMPLETED',
      seed=42,baseline=baseline,e1=new,difference=delta,primary_auc_improved=improved,
      additional_seeds=[43,44] if improved else [],baseline_rerun=False,
      training_epochs=400,new_optimizer_steps=sum(int(r['batches']) for r in rows),
      formal_primary_rollouts=200,formal_secondary_rollouts=680,
      source_and_features_unchanged=True,training=training,disclosed_gaps=contract['disclosed_gaps'],timestamp=time.time())
    write(O/'FINAL_ACCEPTANCE.json',result);write(O/'RUN_STATUS.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
