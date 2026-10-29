"""Compare accepted latest GOAL S with existing full C. No new scientific calls."""
from pathlib import Path
import json,hashlib,statistics,sys
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);W=R.parent
C41=W/'20261025_ccbp_v3_full/final_readonly_20261025_181226'
C=W/'20261027_c_goal42_43_object41_42_43/final_readonly_20261027_222119'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sa=js(D/'LOCAL_FINAL_AUDIT.json');sv=js(D/'EVIDENCE.json');ca=js(C/'LOCAL_FINAL_AUDIT.json');cv=js(C/'EVIDENCE.json');c41a=js(C41/'LOCAL_FINAL_AUDIT.json');c41v=js(C41/'EVIDENCE.json')
assert sa['status']==ca['status']==c41a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert sa['evidence_sha256']==sha(D/'EVIDENCE.json') and ca['source_sha256']==sha(C/'EVIDENCE.json') and c41a['evidence_sha256']==sha(C41/'EVIDENCE.json')
keys=['AUC','FWT','NBT','OldFinal','FinalAll'];results={};paired=[];negative=[]
for seed in ['41','42','43']:
 sr=json.loads(sv['light']['seed'+seed+'/OUTCOMES.json']['text'])
 if seed=='41':
  cr=[x for x in json.loads(c41v['light']['OUTCOMES.json']['text']) if x['arm']=='C'];cm=c41a['metrics']['C'];matrix=c41a['matrix']['C']
 else:
  case='GOAL_'+seed;cr=json.loads(cv['light'][case+'/OUTCOMES.json']['text']);cm=ca['metrics'][case];matrix=ca['matrix'][case]
 index=lambda rows:{(x['stage'],x['task'],x['initial_index']):x for x in rows}
 ss,cc=index(sr),index(cr);assert len(ss)==len(cc)==200 and set(ss)==set(cc)
 for k,x in ss.items():
  y=cc[k];assert x['attempt']==y['attempt']==1
  assert all(x[f]==y[f] for f in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed'])
  paired.append(dict(seed=seed,stage=k[0],task=k[1],initial_index=k[2],C_success=y['success'],S_success=x['success'],initial_sha256=x['initial_sha256'],noise_sha256=x['noise_sha256'],env_seed=x['env_seed']))
 cells={}
 for stage in range(6,10):
  for task in range(6,stage+1):
   rows=[x for x in paired if x['seed']==seed and x['stage']==stage and x['task']==task];cs=sum(x['C_success'] for x in rows);s=sum(x['S_success'] for x in rows);key=f'{stage}_{task}'
   assert cs==matrix[key]['successes'] and s==sa['seeds'][seed]['matrix'][key]['successes']
   cells[key]=dict(C_percent=cs*5,S_percent=s*5,delta_pp=(s-cs)*5,C_success_S_fail=sum(x['C_success'] and not x['S_success'] for x in rows),C_fail_S_success=sum(not x['C_success'] and x['S_success'] for x in rows))
   if s<cs:negative.append(dict(seed=seed,cell=key,**cells[key]))
 sm=sa['seeds'][seed]['metrics'];results[seed]=dict(C_metrics=cm,S_metrics=sm,delta_metrics_pp={k:sm[k]['percent']-cm[k]['percent'] for k in keys},cells=cells)
means={m:{k:statistics.mean(r[m+'_metrics'][k]['percent'] for r in results.values()) for k in keys} for m in ['C','S']}
out=dict(status='PASS_ACCEPTED_C_S_600_EPISODE_PAIRING',primary_reference='Existing complete C GOAL41/42/43; no new C training',sources={str(p):sha(p) for p in [C41/'LOCAL_FINAL_AUDIT.json',C41/'EVIDENCE.json',C/'LOCAL_FINAL_AUDIT.json',C/'EVIDENCE.json',D/'LOCAL_FINAL_AUDIT.json',D/'EVIDENCE.json']},paired_episodes=600,seeds=results,means=means,mean_delta_pp={k:means['S'][k]-means['C'][k] for k in keys},negative_cells=negative,paired_outcomes=paired,costs=dict(C_production_optimizer=228300,S_production_optimizer=426600,C_training_collection=0,S_training_collection=120,formal_episodes_per_method=600),limitations=['Not equal interaction/compute; cannot isolate gate effect','Paired initial bytes/noise do not imply identical visited observations','Three development seeds and 20 episodes per cell do not establish statistical significance','Existing C hardware differs; all negative cells retained'])
(D/'PRIMARY_C_COMPARISON.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['## 主对照：最新完整S与已有完整C GOAL三个seed','','全部600回合初态文件、实际字节、噪声与环境seed逐项配对通过；完整C41来自已接受v3，C42/43来自已接受扩展。原S不替代主对照。','','|指标（三seed均值）|C|最新S|S−C pp|','|---|---:|---:|---:|']
for k in keys:lines.append('|%s|%.4f|%.4f|%+.4f|'%(k,means['C'][k],means['S'][k],out['mean_delta_pp'][k]))
lines+=['','|seed|ΔAUC|ΔFWT|ΔNBT（负值好）|ΔOldFinal|ΔFinalAll|','|---|---:|---:|---:|---:|---:|']
for seed,r in results.items():lines.append('|'+seed+'|'+'|'.join('%+.4f'%r['delta_metrics_pp'][k] for k in keys)+'|')
lines+=['','所有负单元：','','|seed|阶段/任务|C|S|Δpp|C成功S失败|C失败S成功|','|---|---|---:|---:|---:|---:|---:|']
for r in negative:lines.append('|%s|%s|%.1f|%.1f|%+.1f|%d|%d|'%(r['seed'],r['cell'],r['C_percent'],r['S_percent'],r['delta_pp'],r['C_success_S_fail'],r['C_fail_S_success']))
lines+=['','S增加120次训练采集，生产物理调用由C的228300增加至426600；不等交互/算力，不能把差异归因于门控单一机制。须结合分任务SF/FS和旧任务保持/新任务学习解释，不能仅凭最终均值或门控接受率宣称有效。']
(D/'PRIMARY_C_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],means=means,delta=out['mean_delta_pp']),indent=2))
