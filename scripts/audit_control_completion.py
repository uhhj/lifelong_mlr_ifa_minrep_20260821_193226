"""Final read-only validation of complete paired runs and evaluation coverage."""
import csv
import json
from pathlib import Path
from compute_object_metrics import LINE

root=Path(__file__).resolve().parents[1]
expected={(task,99,stage) for stage in range(6,10) for task in range(6,stage+1)}
rows=[]
for seed in (42,43,44):
    old=root/f'runs/final_paper_text_faithful/ifa/seed{seed}'
    new=root/f'runs/own_reference_control/ifa/seed{seed}'
    protocols=[]
    for run in (old,new):
        protocol=list(csv.DictReader((run/'run_001/lr_protocol.csv').open()))
        assert len(protocol)==400,(run,len(protocol))
        protocols.append([{k:v for k,v in row.items() if k!='arm'} for row in protocol])
    assert protocols[0]==protocols[1],f'seed{seed} optimizer/LR protocol differs'
    for arm,run in (('release_ifa',old),('corrected_ifa',new),('mlr',root/f'runs/final_paper_text_faithful/mlr/seed{seed}')):
        assert (run/'PRIMARY_EVAL_COMPLETE').is_file(),run
        records=[m.groups() for m in LINE.finditer((run/'primary_ep99.log').read_text())]
        keys=[tuple(map(int,r[:3])) for r in records]
        assert len(records)==10 and len(set(keys))==10 and set(keys)==expected,(run,keys)
        values=[float(r[3]) for r in records]
        assert all(0<=v<=1 and abs(v*20-round(v*20))<1e-8 for v in values)
        rows.append(dict(seed=seed,arm=arm,matrix_entries=10,unique_expected_entries=True,success_rates_on_20_trial_grid=True))
result=dict(verdict='COMPLETENESS_CHECK_PASS',optimizer_lr_protocol_equal_for_all_1200_epochs=True,coverage=rows,scope='Exact equality of logged batch counts, optimizer step counts and LR trajectory; all9 evaluation matrices have10 unique expected cells. Trial count also relies on executed eval.n_eval=20 and unchanged evaluator, not only the success-rate grid.')
print(json.dumps(result,indent=2))
