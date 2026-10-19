"""Extend existing count checks with exact baseline metrics and actual RA admission/progress."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,runpy,contextlib,io,sys,statistics,datetime
R=Path(__file__).resolve().parent;D=R/sys.argv[1];E=D/'baseline_evidence';RA=D/'ra_evidence'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(R/'audit_progress.py'),run_name='__main__')
a=js(D/'PROGRESS_ACCEPTANCE.json');summary=js(D/'baseline/SUMMARY.json');accept=js(D/'baseline/INDEPENDENT_ACCEPTANCE.json')
assert accept['status']=='PASS' and accept['formal_records']==400 and accept['final_checkpoints']==8
assert len(accept['loaded_downstream_keys'])==8 and set(accept['loaded_downstream_keys'].values())=={100}
for n in ['pipeline.exit','supervisor.exit']:assert (D/'baseline'/n).read_text().strip()=='0'
metrics={};matrices={}
for arm in ['random','coreset']:
 v=a['arms'][arm];assert v['records']==200 and v['vector_attempts_complete']==v['vector_attempts_started']==20
 rates={(x['stage'],x['task']):F(x['successes'],20) for x in v['partial_cells']};assert len(rates)==10
 m=dict(AUC=sum(sum(rates[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(rates[t,t] for t in range(6,10))/4,NBT=sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(rates[9,t] for t in range(6,9))/3,FinalAll=sum(rates[9,t] for t in range(6,10))/4)
 for k in ['AUC','FWT','NBT']:assert m[k]==F(summary['arms'][arm]['metrics'][k]['fraction'])
 matrix=[[float(rates[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)]
 assert matrix==summary['arms'][arm]['matrix'];matrices[arm]=matrix
 metrics[arm]={k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
 v['strict_final_checkpoint_acceptance']='PASS_4_FINAL_CP_100_KEYS_EACH'
proof=js(D/'RA_FROZEN_INPUT_AUDIT.json');assert proof['status']=='PASS_RA_FROZEN_INPUTS'
for n,h in proof['manifest'].items():assert sha(RA/n)==h['sha256']
assert 'Ran 8 tests' in (RA/'admission/software.log').read_text()
for n in ['software','prepare','offline','freeze','preflight']:assert (RA/'admission'/f'{n}.exit').read_text().strip()=='0'
g=js(RA/'GPU_PREFLIGHT.json');assert g['status']=='PASS' and g['isolated_optimizer_calls']==4 and g['formal_rollouts']==g['environment_steps']==0
assert g['contract_sha256']==sha(RA/'CONTRACT.json')
assert js(RA/'ra_coreset/INITIAL_STATE.json')==js(E/'coreset/INITIAL_STATE.json')
ep=js(RA/'ra_coreset/epochs.json');assert len({(x['task'],x['epoch']) for x in ep})==len(ep)
assert all(x['batches']==[100,139,185,337][x['task']-6] for x in ep)
assert sum(x['batches'] for x in ep)==ep[-1]['steps']
sel={str(t):js(RA/'offline_selection'/f'task{t}.json') for t in range(6,10)}
for t,x in sel.items():
 assert len(x['anchor_ids'])==[249,222,226,492][int(t)-6] and set(x['anchor_ids'])<=set(x['selected_ids'])
 assert len(x['selected_ids'])==len(set(x['selected_ids']))==[499,445,452,985][int(t)-6]
 assert x['swap_count']==len(set(x['anchor_ids'])-set(x['original_ids']))==g['selection'][t]['swaps']
baseline_ep=js(E/'coreset/epochs.json');overhead=js(E/'coreset/TRAINING_COMPLETE.json')['seconds_this_invocation']/sum(x['seconds'] for x in baseline_ep)
done={(x['task'],x['epoch']) for x in ep};remain_train=sum(x['seconds'] for x in baseline_ep if (x['task'],x['epoch']) not in done)*overhead
eval_secs=a['mean_vector_seconds']*20
a.update(status='PASS_BASELINES_COMPLETE_RA_PARTIAL',observed_unix=proof['time'],ra_status=js(RA/'RUN_STATUS.json'),ra_epochs=len(ep),ra_steps=ep[-1]['steps'],RA_preflight='PASS',RA_software_tests=8,RA_offline_swaps={t:x['swap_count'] for t,x in sel.items()},RA_initial_rng_tensor_parity=True,RA_task6_final_tensor_parity='PENDING_UNTIL_TASK6_FINISH',total_epochs=800+len(ep),total_steps=152200+ep[-1]['steps'],full_metrics=metrics,matrices=matrices,remaining_RA_training_minutes_estimate=remain_train/60,remaining_RA_eval_minutes_estimate=eval_secs/60,overhead_minutes_assumption=5,remaining_hours_point_estimate=(remain_train+eval_secs+300)/3600)
a['C_minus_R']={k:dict(fraction=str(F(metrics['coreset'][k]['fraction'])-F(metrics['random'][k]['fraction'])),pp=metrics['coreset'][k]['percent']-metrics['random'][k]['percent']) for k in metrics['random']}
(D/'PROGRESS_ACCEPTANCE.json').write_text(json.dumps(a,indent=2)+'\n')
print(json.dumps({k:a[k] for k in ['status','ra_epochs','ra_steps','RA_offline_swaps','full_metrics','C_minus_R','remaining_hours_point_estimate']},indent=2))
