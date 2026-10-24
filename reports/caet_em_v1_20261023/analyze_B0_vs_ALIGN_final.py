"""Existing accepted records only: descriptive comparison, no new model or environment calls."""
import ast,json,hashlib,datetime
from pathlib import Path
from fractions import Fraction
R=Path(__file__).resolve().parent;F=R/'final_CAET_EM_20261024_153414';D=R/'B0_action_diagnostics_20261024_135651';sources={}
def read(p):
 sources[str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf8'))
closure=read(F/'CLOSURE_INDEX.json');assert closure['status']=='PASS_COMPLETE_LOCAL_CLOSURE'
final=read(F/'LOCAL_FINAL_AUDIT.json');assert final['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert hashlib.sha256((F/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest()==closure['sources'][str((F/'LOCAL_FINAL_AUDIT.json').relative_to(R))]
binding=read(D/'REPORT_BINDING.json')
for name in ('SUMMARY.json','PAIRED_HORIZONS.json','LOCAL_AUDIT.json'):
 assert hashlib.sha256((D/name).read_bytes()).hexdigest()==binding[name]
summary=read(D/'SUMMARY.json');pairs=read(D/'PAIRED_HORIZONS.json');ad=read(D/'LOCAL_AUDIT.json')
assert ad['status']=='PASS_LOCAL_B0_ACTION_SCALAR_AUDIT_NOT_CLOSED_LOOP_ACCEPTANCE'
old=read(R.parent/'20261022_caet_v3_tac_eftr_object41/final_ALIGN_seed41_20261022_220252/LOCAL_FINAL_AUDIT.json')
b0={ast.literal_eval(k):Fraction(v,20) for k,v in final['arms']['B0']['success_counts'].items()}
align={(r['stage'],r['task']):Fraction(r['current']) for r in old['matrix']}
decomp=[]
for t in range(6,10):
 x=sum(b0[s,t] for s in range(t,10))/(10-t);y=sum(align[s,t] for s in range(t,10))/(10-t)
 decomp.append(dict(task=t,B0_lifetime_percent=float(x*100),ALIGN_lifetime_percent=float(y*100),AUC_difference_contribution_pp=float((x-y)*25)))
assert abs(sum(x['AUC_difference_contribution_pp'] for x in decomp)+11.770833333333334)<1e-10
diagnostic={}
for t in range(6,10):
 x=summary[f'stage{t}_task{t}_first'];p=pairs[f'stage{t}_task{t}']
 diagnostic[str(t)]=dict(paired_first=p['first_mean_huber'],paired_future3=p['future3_mean_huber'],all_legal_first_count=x['count'],selected_huber=x['statistics']['huber']['mean'],oracle_huber=x['statistics']['oracle_huber']['mean'],non_oracle_fraction=x['selected_not_oracle_mean_fraction'],gripper_error_gt1=x['error_concentration']['gripper_abs_error_gt1_count'],gripper_huber_share=x['error_concentration']['gripper_huber_share'])
 assert p['first_mean_huber']<p['future3_mean_huber']
model=R/'caet_em_exp/model.py';readout=R/'caet_em_exp/readout.py';train=R/'caet_em_exp/train.py'
for p in (model,readout,train):sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
assert 'self.direct=mlp(512,512,4*512)' in model.read_text() and 'return y[:,-1,3],y[:,-1,0]' in readout.read_text()
out=dict(status='PASS_EXISTING_EVIDENCE_SYNTHESIS_NOT_CAUSAL_IDENTIFICATION',time=datetime.datetime.now().astimezone().isoformat(),complete_final_acceptance=True,task_AUC_decomposition=decomp,first_action_diagnostics=diagnostic,source_sha256=sources,new_model_forwards=0,new_optimizer_updates=0,new_environment_steps=0,new_experiments_authorized=False,limitations=['B0 differs from ALIGN in readout, random adapter and four-step supervision together.','First versus later horizon errors are not the counterfactual comparison against a one-step model.','Oracle uses recorded expert target; alternate mode may still be a valid action.','No matched ALIGN GMM diagnostics or new closed-loop state action labels; differential attribution unproven.','Training/RA data, overlapping windows, single seed and repeated benchmark exploration.'])
(R/'B0_VS_ALIGN_FINAL_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],task_AUC_decomposition=decomp,new_forwards=0,new_environments=0),ensure_ascii=False,indent=2))
