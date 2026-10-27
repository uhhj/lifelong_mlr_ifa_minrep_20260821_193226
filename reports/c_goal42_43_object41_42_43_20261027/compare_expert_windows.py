"""Compare accepted C/S scalar diagnostics on identical expert-window identities."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;W=R.parent;D=Path(sys.argv[1]);sources={}
def read(p):
 sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf8'))
comparison=read(D/'COMPARISON.json');assert comparison['status']=='PASS_ACCEPTED_COMPARISON'
old=W/'20261025_ccbp_v3_full/final_readonly_20261025_181226'
sg=W/'20261026_ccbp_v4_s_goal42_43/final_readonly_20261026_164119'
so=W/'20261026_ccbp_v4_s_object41_42_43/final_readonly_20261027_120754'
s41=W/'20261025_ccbp_v4_full/final_readonly_20261026_014654'
def load_diag(root,name):
 ev=read(root/'EVIDENCE.json');p=root/'diagnostics'/name
 meta=next(x for k,x in ev['diagnostic_files'].items() if Path(k).name==name)
 rows=read(p);assert sources[str(p)]==meta['sha256'] and len(rows)==meta['rows'];return rows
def score(row,method):
 c=np.asarray(row['student_component_group_costs' if method=='C' else 'component_group_costs']).sum(-1)
 z=np.asarray(row['student_logits' if method=='C' else 'logits']);k=int(z.argmax());assert k==row['student_mode']
 return dict(expert=float(c[k]),regret=float(c[k]-c.min()),nonmin=k!=int(c.argmin()))
result={};discordant=[]
for suite in ['GOAL','OBJECT']:
 for seed in [41,42,43]:
  case=f'{suite}_{seed}';cr=old if case=='GOAL_41' else D;sr=s41 if case=='GOAL_41' else (sg if suite=='GOAL' else so)
  for stage in [7,8,9]:
   cn=f'C_{stage}.json' if case=='GOAL_41' else f'{case}_C_{stage}.json'
   sn=f'expert_S_{stage}.json' if case=='GOAL_41' else f's{seed}_expert_S_{stage}.json'
   cc=load_diag(cr,cn);ss=load_diag(sr,sn)
   key=lambda r:(r['index'],r['task'],r['sample_sha256'])
   cm={key(x):x for x in cc};sm={key(x):x for x in ss};assert len(cm)==len(cc)==len(sm)==len(ss) and cm.keys()==sm.keys()
   for task in range(6,stage):
    pairs=[(score(cm[k],'C'),score(sm[k],'S')) for k in cm if k[1]==task];assert pairs
    stat={m:{field:float(np.mean([p[i][field] for p in pairs])) for field in ['expert','regret','nonmin']} for i,m in enumerate(['C','S'])}
    cell=comparison['paired_C_minus_S'][case][f'{stage}_{task}']
    entry=dict(windows=len(pairs),C=stat['C'],S=stat['S'],C_minus_S_expert=stat['C']['expert']-stat['S']['expert'],C_minus_S_closed_loop_pp=cell['C_minus_S_pp'],S_success_C_failure=len(cell['S_success_C_failure']),S_failure_C_success=len(cell['S_failure_C_success']))
    result[f'{case}_{stage}_{task}']=entry
    if entry['C_minus_S_expert']*entry['C_minus_S_closed_loop_pp']>0:discordant.append(f'{case}_{stage}_{task}')
out=dict(status='PASS_COMMON_EXPERT_WINDOW_COMPARISON',cells=result,offline_closed_loop_opposite_direction=discordant,source_sha256=sources,limits=['Identical replay windows do not supply matched closed-loop observation histories','Lower expert expected squared error is a replay-state proxy, not trajectory success','C and S also differ in fixed retention and 40 training collection episodes per seed','No policy change or new model/environment execution'])
(D/'MATCHED_EXPERT_ANALYSIS.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(dict(status=out['status'],matched_cells=len(result),discordant_cells=len(discordant),stage9={k:v for k,v in result.items() if k.split('_')[-2]=='9'}),indent=2))
