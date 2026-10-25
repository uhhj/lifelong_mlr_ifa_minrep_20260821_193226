"""Accepted evidence only: contrast common-window proxy and closed-loop changes."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
v=js(D/'EVIDENCE.json');L={n:json.loads(x['text']) for n,x in v['light'].items() if n.endswith('.json')}
P=R.parent/'20261025_ccbp_v3_full/final_readonly_20261025_181226'
common={};reference={}
for stage in range(7,10):
 old=js(P/('diagnostics/C_%d.json'%stage));ids=[(r['index'],r['task'],r['sample_sha256']) for r in old]
 for arm in 'DOS':
  rr=js(D/('diagnostics/expert_%s_%d.json'%(arm,stage)));assert ids==[(r['index'],r['task'],r['sample_sha256']) for r in rr]
 common[str(stage)]=dict(count=len(ids),reference_C_identity_order_payload_exact=True)
 for task in range(6,stage):
  cost=[];regret=[]
  for r in old:
   if r['task']!=task:continue
   cc=np.array(r['student_component_group_costs']).sum(-1);z=np.array(r['student_logits']);k=int(z.argmax());cost.append(float(cc[k]));regret.append(float(cc[k]-cc.min()))
  reference['C_%d_%d'%(stage,task)]=dict(count=len(cost),execution_mean=float(np.mean(cost)),regret_mean=float(np.mean(regret)),source_sha256=sha(P/('diagnostics/C_%d.json'%stage)))
gradient={}
terms=['bc','ifa_weighted','expert_weighted','expert_route_weighted','retain_weighted','retain_route_weighted']
for arm in 'DOS':
 for stage in range(7,10):
  rr=[x for x in L['GRADIENTS_'+arm+'.json'] if x['stage']==stage];assert len(rr)==4
  gradient['%s_%d'%(arm,stage)]={k:dict(median_l2=float(np.median([x[k]['l2'] for x in rr])),median_ratio_to_BC=float(np.median([x[k]['l2']/max(x['bc']['l2'],1e-300) for x in rr])),nonzero_points=sum(x[k]['l2']>1e-12 for x in rr)) for k in terms}
out=dict(status='ACCEPTED_SCALAR_ANALYSIS',acceptance_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),common_expert_windows=common,reference_C_common_costs=reference,gradient_measurements=gradient,own_memory_costs_not_common_distribution=True,additional_forwards=0,additional_updates=0,additional_environments=0)
(D/'MECHANISM_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
lines=['# 机制解释的证据边界','','|比较|阶段/任务|SF：参考成功候选失败|FS：参考失败候选成功|净变化(pp)|','|---|---|---:|---:|---:|']
for pair,cells in a['paired'].items():
 for cell,m in cells.items():lines.append('|%s|%s|%d|%d|%+d|'%(pair,cell,len(m['reference_success_to_candidate_failure']),len(m['reference_failure_to_candidate_success']),m['delta_pp']))
lines+=['','## 共同示范窗口上的专家误差与闭环差值','','|组别/阶段/旧任务|执行专家误差|相对C误差差|闭环相对C(pp)|','|---|---:|---:|---:|']
for arm in 'DOS':
 for stage in range(7,10):
  stats=a['offline_by_stage_task']['expert_%s_%d'%(arm,stage)]
  for task in range(6,stage):
   cost=stats[str(task)]['statistics']['execution_cost']['mean'];delta=cost-reference['C_%d_%d'%(stage,task)]['execution_mean'];closed=a['paired'][arm+'-C']['%d_%d'%(stage,task)]['delta_pp']
   lines.append('|%s/%d/%d|%.8g|%+.8g|%+d|'%(arm,stage,task,cost,delta,closed))
lines+=['','保持缓存来自不同访问历史，不能把不同组缓存上的距离下降当作同分布性能提升。S有效数据量更少或某任务为空，是成功筛选策略的实际组成部分；O/S还产生不同后续策略与状态分布，不能只归因标签。','', '固定任务获得时参考避免滚动覆盖，但可能永久保留早期错误；轨迹成功不是逐步正确性标签。只有配对闭环支持条件通过才有继续研究依据，离线改善或非零梯度不是任务保持保证。','', '梯度仅四个点的范数，没有方向/裁剪/AdamW后的因果分解；没有不可逆失败时刻标签和物理状态干预，不宣布唯一失败原因。所有负单元和早期7/6、晚期9/8都保留。单seed与反复开发初态仅探索，不自动扩展。']
(D/'MECHANISM_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('Accepted scalar analysis saved; no new experiment calls.')
