"""Exercise the complete report path with synthetic, clearly non-scientific data."""
import json
import tempfile
from pathlib import Path
import numpy as np
from gradient_relation_experiment import summarize as s
from cb_ifa_experiment.common import sha,write


def run():
    a=s.metrics([-1.,-1.,1.,1.],[-2.,-1.,1.,2.],[.01]*4)
    assert a['ROC_AUC']==1. and a['PR_average_precision']==1. and a['zero_threshold_precision']==1.
    assert s.metrics([0.,0.],[1.,2.],[.01,.01])['ROC_AUC'] is None
    assert s.metrics([1.],[float('nan')],[.01])=={'n':0}
    original_root=s.ROOT;original_out=s.OUT;original_verify=s.verify_code
    try:
        with tempfile.TemporaryDirectory(prefix='gradient_relation_analysis_test_') as tmp:
            root=Path(tmp);out=root/'reports';out.mkdir()
            s.ROOT=root;s.OUT=out;s.verify_code=lambda:None
            for seed in (45,46,47):
                rows={k:[] for k in ('scores','labels','checks','carrier_checks')}
                rng=np.random.RandomState(seed)
                for i in range(40):
                    task=7 if i<20 else 8;phase='30-69' if i%20<10 else '70-99';point=f's{seed}_i{i}'
                    rows['checks'].append({'point':point,'carrier_restored_exact':True,'native_11_parity':True})
                    for h in (1,5,20):rows['carrier_checks'].append({'point':point,'horizon':h,'matches':{'model':True,'optimizer':True}})
                    for j in range(6,task):
                        cos=float(rng.uniform(-1,1));mixed=float(rng.normal());nnew=3+i%5;score={'point':point,'seed':seed,'task':task,'phase':phase,
                            'old_task':j,'epoch':35+i%20+(40 if phase=='70-99' else 0),'stage_step':i*100,'cosine':cos,
                            'new_dot_old':cos*2,'new_norm':2.,'old_norm':1.,'mixed_dot_old':mixed,'mixed_norm':3.,
                            'n_new':nnew,'n_old_j':10-nnew,'n_batch':10,'lr':1e-5,'gradient_valid':True,
                            'stability_cosines':[cos,-cos,cos*.5]}
                        rows['scores'].append(score)
                        for bank in ('A','B'):
                            for h in (1,5,20):
                                e=-cos*.03+float(rng.normal(scale=.005));f=-cos*.01+float(rng.normal(scale=.005));risk=mixed*.02+float(rng.normal(scale=.005))
                                rows['labels'].append({'point':point,'seed':seed,'task':task,'phase':phase,'old_task':j,'bank':bank,'horizon':h,
                                    'E_new_to_old':e,'E_old_to_new':f,'H_MLR':risk,'old_tolerance':1e-5,'new_tolerance':1e-5,
                                    'old_dot_d11':mixed,'old_dot_d11_minus_d01':-cos+.1,'new_dot_d11_minus_d10':-cos-.1})
                d=root/f'runs/gradient_relation_v1/seed{seed}';d.mkdir(parents=True)
                files={}
                for name,data in rows.items():
                    p=d/(name+'.jsonl');p.write_text(''.join(json.dumps(x)+'\n' for x in data));files[p.name]=sha(p)
                write(d/'COMPLETE.json',{'status':'PASS','points_completed':40,'carrier_source':'new_controlled_3090_from_official',
                    'files':files,'initial_model_sha256':'synthetic_same_initial','native_steps':69300})
            s.run()
            result=json.loads((out/'COMPLETE.json').read_text());assert result['completed_points']==120
            for name in ('raw_prediction_metrics.csv','LOSO_metrics.csv','LOSO_subgroups.csv','incremental_value.csv','FINAL_REPORT.md','FINAL_DELIVERY.zip'):
                assert (out/name).stat().st_size>0
    finally:s.ROOT=original_root;s.OUT=original_out;s.verify_code=original_verify
    write(original_out/'ANALYSIS_TEST.json',{'status':'PASS','synthetic_only':True,'scientific_outcomes_used':False,
        'checks':['known perfect ranking','single-class NA','invalid score missing','full120-origin synthetic report','LOSO seed split','task/phase subgroups','delivery archive']})
    print('ANALYSIS_TEST_PASS')


if __name__=='__main__':run()
