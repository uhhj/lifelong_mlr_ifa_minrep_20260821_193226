"""Explain only independently accepted L005 results. No scientific execution."""
from pathlib import Path
import json,sys,hashlib,datetime,statistics
R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_FINAL_AUDIT.json');c=js(D/'PRIMARY_S_COMPARISON.json');v=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_L005_LOCAL_ACCEPTANCE' and c['paired_episodes']==600
assert c['source_Variant_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json') and a['evidence_sha256']==sha(D/'EVIDENCE.json')
base=js(Path(c['source_S']['L005']['final_path'])/'EVIDENCE.json')
details={};all_losses=[];all_gains=[]
for case,x in c['cases'].items():
 seed=int(case[-2:]);pair=x['paired_outcomes'];cells=x['cells'];groups={}
 for name,rows in [('all',pair),('acquisition',[r for r in pair if r['stage']==r['task']]),('old',[r for r in pair if r['stage']>r['task']]),('final',[r for r in pair if r['stage']==9])]:
  groups[name]=dict(n=len(rows),L005_success_L010_failure=sum(r['Variant'] and not r['S'] for r in rows),L005_failure_L010_success=sum(not r['Variant'] and r['S'] for r in rows),L005_successes=sum(r['Variant'] for r in rows),L010_successes=sum(r['S'] for r in rows))
 losses=[];gains=[]
 for k,row in cells.items():
  out=dict(case=case,cell=k,L005=row['Variant_percent'],L010=row['S_percent'],delta_L005_minus_L010=row['Variant_percent']-row['S_percent'],gain_initials=row['Variant_success_S_fail'],loss_initials=row['Variant_fail_S_success'])
  if out['delta_L005_minus_L010']<0:losses.append(out);all_losses.append(out)
  if out['delta_L005_minus_L010']>0:gains.append(out);all_gains.append(out)
 proxies={}
 for label,ev,prefix in [('L005',v,case+'/'),('L010',base,f'seed{seed}/')]:
  light=ev['light'];train=json.loads(light[prefix+'TRAIN_S.json']['text']);gg=json.loads(light[prefix+'GRADIENTS_S.json']['text'])
  perstage={}
  for t in (7,8,9):
   g=[g for g in gg if g['stage']==t]
   gates=[json.loads(line) for line in light[prefix+f'GATES_stage{t}.jsonl']['text'].splitlines()] if prefix+f'GATES_stage{t}.jsonl' in light else []
   perstage[t]=dict(gradient_samples=len(g),retain_bc_conflicts=sum(row['retain_vs_bc']['dot']<0 for row in g),retain_exec_conflicts=sum(row['retain_vs_exec']['dot']<0 for row in g),last_gradient=g[-1],blocks=len(gates),Q_accepts=sum(row['accepted']=='Q' for row in gates),criterion_passes={k:sum(row['checks'][k] for row in gates) for k in ['bc_new','exec_new','retain_improved']})
  proxies[label]=dict(Q_accepts=train['Q_accepts'],gate_blocks=train['gate_blocks'],Q_acceptance_percent=100*train['Q_accepts']/train['gate_blocks'],stages=perstage,cache={str(t):{k:json.loads(light[prefix+f'MEMORY_S_{t}.json']['text'])[k] for k in ['collection_successes','count']} for t in (6,7,8,9)})
 details[case]=dict(paired_groups=groups,negative_cells=losses,positive_cells=gains,ties=sum(row['S_minus_Variant_pp']==0 for row in cells.values()),proxies=proxies,hardware=x['hardware'])
summary={name:{k:sum(x['paired_groups'][name][k] for x in details.values()) for k in ['n','L005_success_L010_failure','L005_failure_L010_success','L005_successes','L010_successes']} for name in ['all','acquisition','old','final']}
interpret=dict(status='HUMAN_INTERPRETATION_INPUTS_VERIFIED',difference_orientation='L005 minus L010',paired_summary=summary,all_negative_cells=all_losses,all_positive_cells=all_gains,cases=details,sources=dict(local_audit=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison=sha(D/'PRIMARY_S_COMPARISON.json')))
dump(D/'INTERPRETATION_AUDIT.json',interpret)
print(json.dumps(dict(means=c['suites'],paired_summary=summary,negative_cells=all_losses,positive_cells=all_gains,proxies={k:{m:{n:z[n] for n in ['Q_accepts','gate_blocks','Q_acceptance_percent','cache']} for m,z in x['proxies'].items()} for k,x in details.items()}),ensure_ascii=False,indent=2))
