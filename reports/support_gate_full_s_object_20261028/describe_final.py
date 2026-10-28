"""Describe genuinely accepted closed-loop results; no model/environment operations."""
from pathlib import Path
import json,sys,statistics,hashlib
R=Path(__file__).resolve().parent;D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
a=js(D/'LOCAL_FINAL_AUDIT.json');v=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal_episodes']==600
keys=['AUC','FWT','NBT','OldFinal','FinalAll'];seeds=a['seeds'];pairs=[]
for seed,result in seeds.items():
 records=json.loads(v['light']['seed'+seed+'/OUTCOMES.json']['text'])
 for t in range(6,9):
  first={x['initial_index']:x for x in records if x['stage']==x['task']==t}
  last={x['initial_index']:x for x in records if x['stage']==9 and x['task']==t}
  assert set(first)==set(last)==set(range(20))
  for i in first:assert all(first[i][k]==last[i][k] for k in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed'])
  pairs.append(dict(seed=seed,task=t,first_successes=sum(x['success'] for x in first.values()),last_successes=sum(x['success'] for x in last.values()),SF=sum(first[i]['success'] and not last[i]['success'] for i in first),FS=sum(not first[i]['success'] and last[i]['success'] for i in first)))
stats={k:dict(mean=statistics.mean(x['metrics'][k]['percent'] for x in seeds.values()),sample_sd=statistics.stdev(x['metrics'][k]['percent'] for x in seeds.values())) for k in keys}
out=dict(status='ACCEPTED_EXECUTION_SCIENTIFIC_INTERPRETATION_REQUIRED',statistics=stats,within_seed_stage_pairs=pairs,gate_audits={k:x['gate_audit'] for k,x in seeds.items()},final_audit_sha256=hashlib.sha256((D/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest(),limits=a['limits'])
(D/'RESULT_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
lines=['# 最新完整 S OBJECT 三 seed 闭环结果','','独立执行验收通过。下面分别报告闭环表现与训练门控统计；验收通过不代表科学收益已证明。','','|seed|AUC|FWT|NBT（低好）|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for seed in ('41','42','43'):lines.append('|'+seed+'|'+'|'.join('%.4f'%seeds[seed]['metrics'][k]['percent'] for k in keys)+'|')
lines+=['|均值±样本标准差|'+'|'.join('%.4f±%.4f'%(stats[k]['mean'],stats[k]['sample_sd']) for k in keys)+'|','','FWT沿项目定义为任务获得时成功率均值；AUC为每任务后续阶段平均再对任务平均。每cell20回合，一个回合对应5个百分点。','','|seed/阶段|Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|']
for seed in ('41','42','43'):
 for stage in range(6,10):lines.append('|'+seed+'/'+str(stage)+'|'+'|'.join(str(seeds[seed]['matrix'][f'{stage}_{t}']['percent']) if t<=stage else '—' for t in range(6,10))+'|')
lines+=['','|seed/旧任务|获得时成功|最终成功|SF|FS|','|---|---:|---:|---:|---:|']
for p in pairs:lines.append('|%s/%d|%d|%d|%d|%d|'%(p['seed'],p['task'],p['first_successes'],p['last_successes'],p['SF'],p['FS']))
lines+=['','|seed/训练阶段|缓存支持度q|Q接受块/总块|接受率|','|---|---|---:|---:|']
for seed,x in seeds.items():
 for stage,g in x['gate_audit']['by_stage'].items():lines.append('|%s/%s|%s|%d/%d|%.2f%%|'%(seed,stage,json.dumps(g['support']),g['Q_accepts'],g['blocks'],100*g['Q_accepts']/g['blocks']))
lines+=['','门控接受率、训练探针误差和梯度方向仅说明训练过程；须结合上面的新任务获得表现、旧任务SF/FS和最终成功率解释。q是缓存覆盖支持度，不是成功概率。旧机制已由用户取消，不恢复，也不作为本轮验收前置。历史结果仅作只读参考，不能称等算力或等交互因果对照。','', '1200 epoch、316500接受更新、585900实际训练优化器调用；273隔离更新、123准入额外autograd和108方向诊断autograd另记。120训练采集和600正式回合。没有补采、重试、测试选优或自动新增实验。','', '报告发布前仍需人工核对解释文字：逐项说明新旧任务变化与负结果，不能仅凭自动表格宣布机制有效。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
