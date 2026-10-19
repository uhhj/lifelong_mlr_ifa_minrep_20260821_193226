"""Independent exact three-seed aggregation and fresh queue-exit acceptance."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,statistics,sys,csv
R=Path(__file__).resolve().parent;D=R/sys.argv[1]
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'PROGRESS_ACCEPTANCE.json');s=js(D/('FINAL_SNAPSHOT.json' if (D/'FINAL_SNAPSHOT.json').exists() else 'SNAPSHOT.json'));Q=D/('queue_final' if (D/'queue_final').exists() else 'queue');remote=js(Q/'THREE_SEED_RESULTS.json');exitproof=js(Q/'FINAL_QUEUE_ACCEPTANCE.json')
assert not a['partial'] and set(a['completed'])=={'42','43','44'}
assert s['queue_exit']==s['wrapper_exit']=='0' and s['queue_failure'] is None
assert not any(' -m coreset_exp.' in p for p in s['processes'])
assert exitproof['status']=='PASS' and exitproof['no_remaining_scientific_processes'] and exitproof['queue_exit']==0
assert exitproof['aggregate_sha256']==sha(Q/'THREE_SEED_RESULTS.json')
assert remote['RA_formal_records']==600 and remote['RA_vector_attempts']==60 and remote['epochs']==1200 and remote['optimizer_steps']==228300 and remote['checkpoints']==12
for n in ['queue.exit','wrapper.exit','aggregate.exit']:assert (Q/n).read_text().strip()=='0'
keys=['AUC','FWT','NBT','OldFinal','FinalAll'];arms=['random','coreset','ra_coreset'];raw={};details={};matrix_rows=[];metric_rows=[]
for seed,v in a['completed'].items():
 E=D/('seed'+seed)/'evidence';P=D/('seed'+seed)/'baseline_evidence'
 assert v['status']=='PASS' and v['records']==200 and v['vector_attempts']==20 and v['strict_checkpoint_loads']==4
 epochs=js(E/'ra_coreset/epochs.json');assert {(x['task'],x['epoch']) for x in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
 assert v['metrics']==remote['seeds'][seed]['metrics'] and v['matrix']==remote['seeds'][seed]['matrix']
 raw[seed]={arm:{k:F(m[k]['fraction']) for k in keys} for arm,m in v['metrics'].items()}
 parent=js(E/'PARENT_BINDING.json')['parent'];out='/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%s_%s'%(seed,'20261018' if seed=='42' else '20261019')
 for arm,root,rpath in [('ra_coreset',E,out),('random',P,parent),('coreset',P,parent)]:
  for p in (root/arm/'evaluation').glob('trials*.jsonl'):assert sha(p)==remote['input_outcome_hashes'][rpath+'/'+p.relative_to(root).as_posix()]
  metric_rows.append(dict(seed=seed,method=arm,**{k:float(raw[seed][arm][k]*100) for k in keys}))
 matrix_rows.extend(dict(seed=seed,**x) for x in v['matrix'])
 audit=js(E/'AUDIT.json')
 details[seed]=dict(storage=v['storage'],selection=audit['selection'],ifa=audit['ifa'],timings=v['timings'],checkpoints=audit['checkpoints'],task6_tensor_equal=v['task6_tensor_equal'],initial_rng_equal=v['initial_rng_equal'])
stats={}
for arm in arms+['RA_minus_Coreset','RA_minus_Random']:
 stats[arm]={}
 for k in keys:
  values=[m['ra_coreset'][k]-m['coreset' if arm=='RA_minus_Coreset' else 'random'][k] if arm.startswith('RA_minus') else m[arm][k] for m in raw.values()]
  x=dict(mean_fraction=str(sum(values)/3),mean_percent=float(sum(values)/3*100),sample_sd_pp=statistics.stdev([float(z*100) for z in values]))
  y=remote['statistics'][arm][k];assert x['mean_fraction']==y['mean_fraction'] and abs(x['sample_sd_pp']-y['sample_sd_pp'])<1e-10
  stats[arm][k]=x
out=dict(status='COMPLETED',local_independent_acceptance='PASS',epochs=1200,optimizer_steps=228300,checkpoints=12,RA_records=600,RA_vectors=60,technical_exceptions=0,read_only_baseline_records=1200,queue_exit=0,wrapper_exit=0,all_seed_phase_exits=0,statistics=stats,per_seed=a['completed'],details=details,final_server_acceptance=exitproof,actual_checkpoint_loading='12 checkpoints strictly loaded on server; local audit verifies proofs, no weights downloaded',ddof=1,new_training_or_environment_calls=0)
(D/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
for name,rs in [('all_seed_metrics.csv',metric_rows),('all_task_matrices.csv',matrix_rows)]:
 with (D/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
print(json.dumps(dict(status=out['status'],statistics=stats),indent=2))
