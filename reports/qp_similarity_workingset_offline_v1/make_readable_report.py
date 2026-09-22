import csv,json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
s=json.loads((P/'SUMMARY.json').read_text());rows=list(csv.DictReader((P/'per_instance.csv').open()))
primary=[r for r in rows if float(r['tau'])==.999 and r['arm']=='SIM_ADD']
f=lambda x:'N/A' if x is None else ('%.4g'%float(x))
lines=['# 相似度引导约束工作集：离线可行性实验 v1','',
'主结论：`'+s['primary_label']+'`。主阈值固定为0.999；0.99/0.9999不替代主结论。','',
'## 四项假设','',
'| 指标 | 结果 |','|---|---|']
for k,v in s['gates'].items():lines.append('| '+k+' | '+str(v)+' |')
lines+=['','## 五个真实压力实例（全部64约束）','',
'| 实例 | 双精度KKT认证及修正预算 | k0→k最终 | 最终非零支撑 | 峰值线性维度 | 子QP轮数 | E_update | SIM/FULL总时间 | SIM/随机中位总时间 |',
'|---|---|---|---|---|---|---|---|---|']
ratios={r['fixture']:r for r in s['real_stress']['per_fixture']}
for r in primary:
 if r['source']!='real':continue
 q=ratios[r['fixture']]
 lines.append('| '+' | '.join([r['fixture'],r['certified'],r['k0']+'→'+r['k_final'],r['final_nonzero_support'],r['peak_linear_dimension'],r['subqp_rounds'],f(r['E_update']),f(q['T_SIM_FULL']),f(q['T_SIM_RANDOM'])])+' |')
lines+=['','真实五例SIM/FULL时间比几何平均：'+f(s['real_stress']['geometric_mean_T_SIM_FULL'])+'；最慢退化比：'+f(s['real_stress']['slowest_T_SIM_FULL'])+'。','',
'## 合成数据与确认集','',
'| 集合 | 数量 | 最终约束占比中位数 | SIM/FULL时间中位比 | SIM/随机时间中位比 | SIM/随机线性求解次数中位比 | 相似度增量H3 |',
'|---|---|---|---|---|---|---|']
for name,g in [('全部高冗余/混合',s['high_mixed']),('全部低冗余',s['low'])]+list(s['subsets'].items()):
 lines.append('| '+' | '.join([name,str(g['n']),f(g.get('median_k_final_fraction')),f(g.get('median_T_SIM_FULL')),f(g.get('median_T_SIM_RANDOM')),f(g.get('median_solves_SIM_RANDOM')),str(g.get('H3'))])+' |')
lines+=['','## 数值与机制','']
lines.append('主版本完整认证通过 '+str(sum(r['certified']=='True' for r in primary))+'/'+str(len(primary))+'；所有失败均在per_instance.csv保留。')
ones=[r for r in rows if float(r['tau'])==.999 and r['arm']=='SIM_ONESHOT']
lines.append('仅首轮不补回：'+str(sum(r['full_kkt_pass']=='True' for r in ones))+'/'+str(len(ones))+' 通过全量KKT。此比例不等于机器人性能。')
values=[float(r['E_update']) for r in primary if r['E_update']]
lines.append('主版本最大E_update：'+f(max(values) if values else None)+'；工程预算1e-5。')
lines+=['','## 固定敏感性（描述性，不替代主结果）','','| 阈值 | 完整认证 | 高冗余/混合最终候选占比 | 高冗余/混合SIM/FULL时间比 | 高冗余/混合SIM/随机时间比 |','|---|---|---|---|---|']
for tau in [.999,.99,.9999]:
 sims=[r for r in rows if float(r['tau'])==tau and r['arm']=='SIM_ADD'];rf=[];rr=[];fractions=[]
 for sim in sims:
  if sim['kind'] not in ['high','mixed']:continue
  same=[r for r in rows if float(r['tau'])==tau and r['fixture']==sim['fixture']]
  full=next(r for r in same if r['arm']=='FULL_WS');random=[r for r in same if r['arm']=='RANDOM_ADD']
  rf.append(float(sim['total_seconds_median'])/float(full['total_seconds_median']))
  rr.append(float(sim['total_seconds_median'])/statistics.median(float(r['total_seconds_median']) for r in random))
  fractions.append(float(sim['k_final'])/float(sim['m']))
 lines.append('| '+' | '.join([str(tau),str(sum(r['certified']=='True' for r in sims))+'/'+str(len(sims)),f(statistics.median(fractions)),f(statistics.median(rf)),f(statistics.median(rr))])+' |')
lines+=['','## 审计与限制','',
'- 主程序执行前冻结；确认seed2026092203在冻结实现后运行，未据结果改变参数。',
'- 各分支预热1次、计时10次并交错；重复次数不是独立实例数。',
'- 初始推送期间可能存在CPU竞争，TIMING_ISOLATION_NOTE中事先登记的全部单元统一重测；旧计时保留在archive_preserved_timing，不择优取时。',
'- 工作集内核和Decimal80验收复用原实现；参考解仅用于评估，未给SIM或随机分支初值。',
'- 总时间包括算法内部精度复核和原内核条件数/trace开销；不包括输入输出、评估参考制作和额外报告。',
'- RANDOM_ADD三种子逐项报告，归因比较使用三者中位数。',
'- 五个真实难例是压力集，不代表训练总体；真实普通32/96样本未新增。',
'- 无GPU、无模型写回、无逐样本反向传播或闭环验证，不能外推整体训练提速或成功率。',
'- 精确重复/奇异边界允许明确UNRESOLVED，不能据此宣称未来退化QP均可解。','',
'`training_active=false; formal_solver_modified=false; model_writeback=false; GPU_required=false`','']
(P/'RESULTS_ZH.md').write_text('\n'.join(lines))
