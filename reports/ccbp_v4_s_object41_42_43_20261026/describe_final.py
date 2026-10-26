"""Describe accepted OBJECT S three-seed results; no forwards or environments."""
import json,sys,hashlib,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
v=js(D/'EVIDENCE.json');ref=js(R/'REFERENCE_RESULTS.json')
for name,h in ref['source_sha256'].items():assert sha(R.parents[1]/name)==h
S=a['seeds'];keys=['AUC','FWT','NBT','OldFinal','FinalAll'];stats={k:dict(mean=statistics.mean(x['metrics'][k]['percent'] for x in S.values()),sample_sd=statistics.stdev(x['metrics'][k]['percent'] for x in S.values())) for k in keys}
base=ref['baselines']['runs'];deltas={};negatives=[];base_stats={};pairs=[]
for seed,ss in S.items():
 deltas[seed]={}
 for rr in [x for x in base if x['seed']==int(seed)]:
  cells={k:ss['matrix'][k]['percent']-x for k,x in rr['matrix'].items()};dm={k:ss['metrics'][k]['percent']-x['percent'] for k,x in rr['metrics'].items()};deltas[seed][rr['method']]=dict(metric_delta_pp=dm,cell_delta_pp=cells,causal_paired_comparison=False)
  negatives.extend(dict(seed=seed,reference=rr['method'],cell=k,delta_pp=x) for k,x in cells.items() if x<0)
 records=json.loads(v['light']['seed'+seed+'/OUTCOMES.json']['text'])
 for t in range(6,9):
  first={x['initial_index']:x for x in records if x['stage']==x['task']==t};last={x['initial_index']:x for x in records if x['stage']==9 and x['task']==t};assert set(first)==set(last)==set(range(20))
  for i in first:assert all(first[i][k]==last[i][k] for k in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed'])
  pairs.append(dict(seed=seed,task=t,initial_success=sum(x['success'] for x in first.values()),final_success=sum(x['success'] for x in last.values()),SF=sum(first[i]['success'] and not last[i]['success'] for i in first),FS=sum(not first[i]['success'] and last[i]['success'] for i in first)))
for method in ['MLR_IFA','RA','RA_ALIGN']:
 rows=[x for x in base if x['method']==method];assert len(rows)==3
 base_stats[method]={k:dict(mean=statistics.mean(x['metrics'][k]['percent'] for x in rows),sample_sd=statistics.stdev(x['metrics'][k]['percent'] for x in rows)) for k in keys}
out=dict(status='ACCEPTED_FINITE_EXTENSION_NO_AUTOMATIC_EXPANSION',suite='OBJECT',S=S,three_seed_statistics=stats,baseline_statistics=base_stats,historical_comparisons=deltas,all_negative_cells=negatives,within_seed_stage_pairs=pairs,final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),reference_sha256=sha(R/'REFERENCE_RESULTS.json'),limits=a['limits'])
(D/'COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
lines=['# S OBJECT seed41/42/43最终结果','','完整有限预算及独立验收通过；这不等于科学优势已获确认。无自动扩seed或调参。','','|seed|AUC|FWT|NBT（低好）|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for seed in ('41','42','43'):lines.append('|'+seed+'|'+'|'.join('%.4f'%S[seed]['metrics'][k]['percent'] for k in keys)+'|')
lines+=['|均值±样本标准差|'+'|'.join('%.4f±%.4f'%(stats[k]['mean'],stats[k]['sample_sd']) for k in keys)+'|','','FWT沿项目定义为任务获得时成功率均值；AUC为各任务后续阶段成功率均值再对任务平均。每cell20回合，5pp为一个回合。','','## 完整矩阵','','|seed/阶段|Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|']
for seed in ('41','42','43'):
 for s in range(6,10):lines.append('|'+seed+'/'+str(s)+'|'+'|'.join(str(S[seed]['matrix'][f'{s}_{t}']['percent']) if t<=s else '—' for t in range(6,10))+'|')
lines+=['','## 同seed历史基线差值','','历史基线非同动作噪声、非等交互，不构成成功筛选的独立因果实验。没有C/O OBJECT三seed对照。原MLR+IFA为项目复现，不是论文表格。','','|seed/基线|ΔAUC|ΔFWT|ΔNBT（负为改善）|ΔOldFinal|ΔFinalAll|','|---|---:|---:|---:|---:|---:|']
for seed in ('41','42','43'):
 for method,x in deltas[seed].items():lines.append('|'+seed+'/'+method+'|'+'|'.join('%+.4f'%x['metric_delta_pp'][k] for k in keys)+'|')
lines+=['','三seed历史基线均值±样本标准差：','','|方法|AUC|FWT|NBT|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for method,st in base_stats.items():lines.append('|'+method+'|'+'|'.join('%.4f±%.4f'%(st[k]['mean'],st[k]['sample_sd']) for k in keys)+'|')
lines+=['','所有负单元：','','|seed|基线|阶段_任务|差值pp|','|---|---|---|---:|']
for n in negatives:lines.append('|%s|%s|%s|%+.1f|'%(n['seed'],n['reference'],n['cell'],n['delta_pp']))
lines+=['','## 本S跨阶段配对变化','','在同seed/任务/初态字节/动作噪声下比较获得阶段与最终阶段，SF为成功转失败，FS反之；这是全轨迹比较，不是相同历史下单步因果干预。','','|seed|任务|获得成功数|最终成功数|SF|FS|','|---|---:|---:|---:|---:|---:|']
for n in pairs:lines.append('|%s|%d|%d|%d|%d|%d|'%(n['seed'],n['task'],n['initial_success'],n['final_success'],n['SF'],n['FS']))
lines+=['','## 最终Stage9离线诊断','','专家误差来自本seed原RA/ALIGN合法窗口，保持距离来自本seed成功采集缓存，两者人口不同。不能用低误差代替闭环，也不能跨seed将不同窗口均值作为因果。','','|seed|类型|任务|人口|执行成本均值|regret均值|非最低成本数|','|---|---|---:|---:|---:|---:|---:|']
for seed in ('41','42','43'):
 for kind in ('expert','retain'):
  for task,x in S[seed]['offline'][f's{seed}_{kind}_S_9'].items():
   st=x['statistics'];lines.append('|%s|%s|%s|%d|%.8f|%.8f|%d|'%(seed,kind,task,x['count'],st['execution_cost']['mean'],st['regret']['mean'],x['nonbest_count']))
lines+=['','## 计算与限制','','三seed1200epoch316500主更新、120训练采集600正式回合；66隔离优化器调用、216正式额外autograd、78准入autograd另列。Task9采集无后续训练用途仍计成本。','','|seed|纯epoch分钟|采集报告分钟|Task6/7/8/9缓存数|最终CPU调用/窗口/秒|','|---|---:|---:|---|---|']
for seed,x in S.items():
 cs=x['collection_cost'];cpu=x['final_CPU'];lines.append('|%s|%.6f|%.6f|%s|%d/%d/%.6f|'%(seed,x['epoch_seconds']/60,sum(y['seconds'] for y in cs.values())/60,'/'.join(str(cs[str(t)]['count']) for t in range(6,10)),cpu['calls'],cpu['windows'],cpu['seconds']))
lines+=['','纯epoch不含保存/准入/采集/缓存/评估/最终诊断；非等算力或等交互比较。成功轨迹筛选不保证每个动作最优，固定获得时教师可能保留不足，缓存不能覆盖所有未来访问历史。没有原始相机或物理状态重演保证，不以本轮标量归因单步失败。保留所有科学负结果，无补采无测试重试；三seed、20回合/cell、开发初态和方法选择限制，不称统计显著。完整有限授权结束后停止。','',*a['limits']]
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],statistics=stats),indent=2))
