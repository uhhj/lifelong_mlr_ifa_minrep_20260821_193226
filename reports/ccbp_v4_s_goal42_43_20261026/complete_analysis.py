"""Describe already accepted scalar evidence; zero model or environment calls."""
import json,sys,statistics,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent; D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
a=js(D/'LOCAL_FINAL_AUDIT.json'); c=js(D/'COMPARISON.json'); ref=js(R/'REFERENCE_RESULTS.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
for name,h in ref['source_sha256'].items():
 assert hashlib.sha256((R.parents[1]/name).read_bytes()).hexdigest()==h
keys=['AUC','FWT','NBT','OldFinal','FinalAll']; lines=[(D/'REPORT.md').read_text(encoding='utf8')]
lines+=['\n## 全部闭环矩阵','每格20次正式回合，单位%。行是训练结束阶段，列是评估任务。训练采集不计入这些结果。','|seed/阶段|Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|']
for seed in ('41','42','43'):
 for s in range(6,10):
  lines.append('|'+seed+'/'+str(s)+'|'+'|'.join(str(c['S'][seed]['matrix'][f'{s}_{t}']['percent']) if t<=s else '—' for t in range(6,10))+'|')
base_stats={}
lines+=['\n## 三种子历史基线对照','下面为均值±样本标准差，NBT低好。原MLR+IFA指项目复现，不是论文表格数值；RA为RA-Coreset+IFA；RA+ALIGN没有C的额外专家损失。','|方法|AUC|FWT|NBT|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for method in ['MLR_IFA','RA','RA_ALIGN']:
 rows=[x for x in ref['baselines']['runs'] if x['method']==method];assert len(rows)==3
 st={k:dict(mean=statistics.mean(x['metrics'][k]['percent'] for x in rows),sample_sd=statistics.stdev(x['metrics'][k]['percent'] for x in rows)) for k in keys};base_stats[method]=st
 lines.append('|'+method+'|'+'|'.join('%.4f±%.4f'%(st[k]['mean'],st[k]['sample_sd']) for k in keys)+'|')
st=c['three_seed_statistics'];lines.append('|S|'+'|'.join('%.4f±%.4f'%(st[k]['mean'],st[k]['sample_sd']) for k in keys)+'|')
lines+=['\n## 逐seed差值与全部负单元','历史方法不是相同动作噪声、相同额外交互的受控对照；不把下表当作成功筛选的独立因果效应。','|seed/历史对照|ΔAUC|ΔFWT|ΔNBT（负为改善）|ΔOldFinal|ΔFinalAll|','|---|---:|---:|---:|---:|---:|']
for seed in ('41','42','43'):
 for method,diff in c['historical_comparisons'][seed].items():
  lines.append('|'+seed+'/'+method+'|'+'|'.join('%+.4f'%diff['metric_delta_pp'][k] for k in keys)+'|')
lines+=['','|seed|对照|阶段_任务|成功率差（pp）|','|---|---|---|---:|']
for n in c['all_negative_cells']:lines.append('|%s|%s|%s|%+.1f|'%(n['seed'],n['reference'],n['cell'],n['delta_pp']))
lines+=['\n## 离线诊断与闭环的对应','专家误差在各seed自身RA/ALIGN合法专家窗口上计算；保持距离在各seed成功采集缓存上计算。两种人口不同，各seed缓存也不同，不能把数值高低直接解释为跨seed因果。下表列最终Stage9；距离为实际argmax执行Normal的分组归一化平方距离。','|seed|窗口类型|任务|窗口数|执行误差/距离均值|选择regret均值|非最低成本数|','|---|---|---:|---:|---:|---:|---:|']
for seed in ('42','43'):
 for kind in ('expert','retain'):
  for task,v in a['seeds'][seed]['offline'][f's{seed}_{kind}_S_9'].items():
   q=v['statistics'];lines.append('|%s|%s|%s|%s|%.8f|%.8f|%s|'%(seed,kind,task,v['count'],q['execution_cost']['mean'],q['regret']['mean'],v['nonbest_count']))
lines+=['','保持缓存重放通过，证明训练使用的固定教师和实际历史绑定正确；它不证明历史覆盖了所有将来失败状态，也不证明成功轨迹内每个动作都是必要或最优的。无新C42/43、O42/43，因此本轮不能分离持续专家监督、固定教师、在线历史分布与成功筛选各自的效应。不得用离线误差或低保持距离替代闭环成功率。','\n## 完整成本与验收','两seed共800完整epoch、152200主更新、80训练采集、400正式回合；44隔离GPU优化器调用、144正式额外autograd、52准入autograd另列。每seedTask9采集缓存没有后续训练用途，仍完整计费。','|seed|纯epoch分钟|采集分钟|保留窗口Task6/7/8/9|最终CPU调用/窗口/秒|','|---|---:|---:|---|---|']
for seed in ('42','43'):
 v=a['seeds'][seed];cs=v['collection_cost'];cpu=v['final_CPU']
 lines.append('|%s|%.6f|%.6f|%s|%d / %d / %.6f|'%(seed,v['epoch_seconds']/60,sum(x['seconds'] for x in cs.values())/60,'/'.join(str(cs[str(t)]['count']) for t in range(6,10)),cpu['calls'],cpu['windows'],cpu['seconds']))
lines+=['','采集分钟取各MEMORY报告的seconds；此前进度报告另用COLLECTION_EVENTS的开始结束差，二者分别保留，不要求跨计时边界完全相同。以上纯epoch不是墙钟总时间；准入、缓存、保存、正式评估和最终CPU诊断分开计数。8个checkpoint实际CPU strict加载、2完整恢复态、8固定教师缓存重放/抽样、80训练与400正式回合初态字节和独立噪声SHA均已真实验收。最终审计零优化器更新、零新增环境；没有补采或失败重试。']
ev=js(D/'EVIDENCE.json');pairs=[]
lines+=['\n## 同一S策略跨阶段的配对变化','只在同seed、同任务、相同初态字节及逐轨迹噪声下比较任务获得时与最终阶段；状态随后可以分化，这是整条闭环结果的比较，不是同历史单步决策干预。SF为获得时成功而最终失败，FS相反。','|seed|任务|获得成功数|最终成功数|SF|FS|','|---|---:|---:|---:|---:|---:|']
for seed in ('42','43'):
 records=json.loads(ev['seeds'][seed]['light']['OUTCOMES.json']['text'])
 for t in range(6,9):
  first={x['initial_index']:x for x in records if x['stage']==x['task']==t};last={x['initial_index']:x for x in records if x['stage']==9 and x['task']==t}
  assert set(first)==set(last)==set(range(20))
  for i in first:
   assert all(first[i][k]==last[i][k] for k in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed'])
  n=dict(seed=seed,task=t,initial_success=sum(x['success'] for x in first.values()),final_success=sum(x['success'] for x in last.values()),SF=sum(first[i]['success'] and not last[i]['success'] for i in first),FS=sum(not first[i]['success'] and last[i]['success'] for i in first));pairs.append(n)
  lines.append('|%s|%d|%d|%d|%d|%d|'%(seed,t,n['initial_success'],n['final_success'],n['SF'],n['FS']))
lines+=['\n## 结果解释','三seed平均AUC相对原MLR+IFA、RA、RA+ALIGN分别高6.42、0.80、1.67pp；OldFinal分别高3.33、2.78、7.22pp。相对RA的平均优势很小，且seed42/43的AUC均低于RA，三seed平均优势依赖seed41。不能概括为各seed都稳定提升。','seed42主要问题包含Task8的获得不足（8_8=60%）和Task9学习不足（75%），不只是后续遗忘：Task8从60%到55%只下降5pp。它的Task6从70%提高到85%，使OldFinal=80%、NBT为负；较低遗忘指标不能掩盖较差的新技能获得。seed43最终Task8保持70%、Task9为95%，但Task7曾在Stage8降到80%后恢复100%，最终均值会漏掉这次中途退化。','两个新seed的Task8缓存都是102窗口，却得到不同最终Task8成功率（55%和70%）。seed42最终Task8专家执行误差约0.00394，低于seed43的0.01546，闭环反而更低。这只说明不同人口下的回放误差不能直接排序闭环性能，不证明较低误差导致失败；也不能把缺额数量单独认定为原因。','设计上S固定了获得时教师、保护成功历史，并没有解决所有未来状态覆盖问题；随机抽样窗口不区分关键接触与普通移动，成功轨迹也可能含冗余或次优动作，固定教师可能保留未充分学好的技能。本轮支持继续重视这些限制，但没有新的受控C/O对照来证明其中哪一个导致退化，不再据此自动开展新实验。','\n## 解释边界与停止决定','本轮是已开发方法在用户指定seed42/43上的有限扩展，不是重新选参。完整保留seed41相对C未通过早期7_6门的事实；seed42/43不存在C门结果，不事后设有利门槛。三seed及旧初态均有开发使用背景，每cell20回合，5个百分点对应一个回合，不声称统计显著或普遍保护保证。硬件差异按授权记录但不作为淘汰条件。全部有限预算已完成，不自动增加种子、消融、阈值或权重扫描。']
(D/'WITHIN_SEED_STAGE_PAIRS.json').write_text(json.dumps(pairs,indent=2)+'\n',encoding='utf8')
# Keep generated Markdown tables and headings separated for the desktop renderer.
for i in range(len(lines)-1,0,-1):
 if lines[i].startswith('|') and not lines[i-1].startswith('|') and lines[i-1]!='':lines.insert(i,'')
 if lines[i].startswith('##') and lines[i+1:i+2]!=['']:lines.insert(i+1,'')
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(D/'BASELINE_STATISTICS.json').write_text(json.dumps(base_stats,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='DESCRIPTIVE_ANALYSIS_COMPLETE',baseline_statistics=base_stats),indent=2))
