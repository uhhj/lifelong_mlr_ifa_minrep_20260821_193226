"""Read-only inventory of hardware recorded for the accepted experiment groups."""
from pathlib import Path
import json,hashlib
A=Path(__file__).resolve().parent;E=A.parents[1]
groups={
'OBJECT full S':'20261028_support_gate_full_s_object/final_readonly_20261029_201235/EVIDENCE.json',
'GOAL full S':'20261029_support_gate_full_s_goal/final_readonly_20261030_111546/EVIDENCE.json',
'OBJECT+GOAL budget C':'20261030_equal_optimizer_budget_cs_two_suites/final_readonly_20261031_065009/EVIDENCE.json',
'OBJECT+GOAL Q':'20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/EVIDENCE.json',
'GOAL lambda .05/.2':'20261103_lambda_r_goal_sensitivity/final_readonly_20261104_122613/EVIDENCE.json',
'OBJECT fixed q1':'20261108_object_s_q1_ablation/final_readonly_20261108_223300/EVIDENCE.json',
'GOAL fixed q1':'20261108_goal_s_q1_ablation/final_readonly_20261109_120106/EVIDENCE.json'}
out={}
for label,rel in groups.items():
 p=E/rel;d=json.loads(p.read_text(encoding='utf8'));rows={}
 for name,item in d['light'].items():
  if name.endswith(('INITIAL_S.json','INITIAL_C.json','INITIAL_Q.json')):
   assert hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256']
   rows[name]=json.loads(item['text'])['hardware']
 assert rows
 out[label]=dict(source=rel,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cases=rows)
p=E/'20261101_lambda_r_object_sensitivity/FINAL_REPORT.md'
out['OBJECT lambda .05/.2 migration']=dict(source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),note='L005_41 first260 complete epochs on4080SUPER; authorized restoration/resume on3090; remaining five runs on3090. Initialization alone cannot describe this resumed run.')
(A/'HARDWARE_INVENTORY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for k,v in out.items():print(k,sorted({r['device'] for r in v.get('cases',{}).values()}) or v['note'])
