import json,statistics,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
r=read(P/'RESULT_REVIEW.json');c=read(P/'coverage.json');q=read(P/'rng_review.json')
root=P.parent/'20261013_coreset_extension/final_evidence/remote'
hashes=read(P/'coverage_input_hashes.json')
for s in (42,43,44):
 contract=read(root/f'seed{s}/CONTRACT.json')
 key=f'/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed{s}_20261013/source/libero/lifelong/selectors/greedy_kcenter.py'
 assert hashes[key]==contract['source_sha256']['libero/lifelong/selectors/greedy_kcenter.py']
 fm=read(root/f'seed{s}/FEATURES.json')
 for f,h in fm['hashes'].items():assert hashes[f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{s}_20261013/features/{f}']==h
assert (P/'coverage.exit').read_text().strip()=='0'
assert c['status']=='PASS' and c['inputs_before_after_unchanged'] and q['inputs_unchanged']
names={6:'黄油',7:'牛奶',8:'巧克力布丁',9:'橙汁'}
lines=['# Random vs Greedy Coreset：只读结果复核与机制分析','',
'复核结论：已记录的结果在本轮既定协议下可信；“Coreset 在这三个 seed 的平均 AUC 更高”成立。尚不足以宣称普遍、稳定、统计显著的优势，也不能把全部收益隔离归因为覆盖距离。此次仅分析既有数据，无训练、环境执行、追加评估或旧产物修改。','',
'## 1. 证据可靠性','',
'- 本地402份证据文件前后SHA256与原发布清单一致。重新从1200条逐episode记录复算全部矩阵与精确分数，吻合原汇总；120个10路环境尝试与记录一致，无技术异常/重试账目。',
'- 核对2400个epoch、633000个optimizer step，以及原独立验收中的24个最终checkpoint严格加载结果。本轮未重新加载模型权重；模型张量一致性依据已保存且哈希匹配的逐张量比较记录，而非checkpoint整文件哈希。',
'- 三个seed的两arm初始化模型/RNG记录相同，阶段6实际模型张量完全一致；阶段6的60对对应初态成功/失败结果也逐条一致。它是入库干预前的重要一致性对照。',
'- 解析后的训练配置差异只有selector与arm输出/缓存路径；实际评估配置差异只有输出路径。两组state=True、预算、损失、优化与最终epoch99规则相同。',
'- 远端CPU只读检查35,572个相关文件（含17,763个原特征、两arm缓存副本、原演示与清单）的分析前后哈希；通过。覆盖复算使用原冻结距离代码，其哈希与原合同一致；所有特征哈希与本地发布清单一致。',
'- 输入源：execution/20261013_coreset_extension/final_evidence/remote/seed{42,43,44}；远端 /data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{42,43,44}_20261013。固定原始commit a721807f2f7fa7f9833dcc85c592d78e9accd00b；实际运行补丁以各seed CONTRACT.json为准。','',
'## 2. 指标复算','',
'单位为百分数/百分点；NBT越低越好。以下AUC按每任务自身已见阶段先平均，再四任务等权平均，不是将所有episode直接混合。','',
'| seed | Random AUC | Coreset AUC | ΔAUC | ΔFWT | ΔNBT |','|---|---:|---:|---:|---:|---:|']
for s,v in r['seeds'].items():
 lines.append(f"| {s} | {v['arms']['random']['metrics']['AUC']:.6f} | {v['arms']['coreset']['metrics']['AUC']:.6f} | {v['delta_pp']['AUC']:+.6f} | {v['delta_pp']['FWT']:+.6f} | {v['delta_pp']['NBT']:+.6f} |")
lines += ['', '三seed平均AUC：Random 76.354167，Coreset 82.916667，配对提升6.562500 ± 7.848528（样本SD，ddof=1）。平均ΔFWT=+2.916667，平均ΔNBT=−6.203704。',
'seed43占三个seed总提升的79.37%；去掉seed43后平均提升仅2.03125个百分点。这是敏感性描述，不是剔除seed43后的主结果。三个seed全部正向，但n=3；仅作符号对称敏感性计算时双侧符号翻转p=0.25。扩展43/44是在看到42之后授权，不能包装成预注册确认性检验。','',
'## 3. 提升来自哪里','',
'表中数值是各任务对总AUC差值的贡献，四行相加才是总提升；不是各任务原始成功率差。','',
'| 任务 | seed42贡献 | seed43贡献 | seed44贡献 | 平均贡献 |','|---|---:|---:|---:|---:|']
for t in range(6,10):
 vals=[r['seeds'][str(s)]['task_AUC_contribution_pp'][str(t)] for s in (42,43,44)]
 lines.append(f'| {t} {names[t]} | {vals[0]:+.6f} | {vals[1]:+.6f} | {vals[2]:+.6f} | {statistics.mean(vals):+.6f} |')
lines += ['',
'- 任务6贡献约46.0%的平均净提升。seed43学完任务7/8/9后，黄油成功率分别由55/40/55%提高到95/85/90%；初学完任务6时两组均60%。这支持后续学习过程中旧任务保持/改善更好，而非初始模型起点更好。',
'- 任务9贡献约31.7%。它仅在最终阶段测一次，属于新任务学习表现：seed43为80→100%，seed44为90→95%，seed42仍45%。不能称为任务9抗遗忘，更不能用训练全部结束后才选出的任务9缓存解释它。',
'- 任务7平均正向，但seed42是负向；任务8每个seed都仅在初学后95→100%，即20次测试多成功1次，最终两组均100%。天花板与小评估样本限制解释力度。',
'- 按AUC权重拆分：初学对角线贡献+2.569444，后续阶段旧任务单元贡献+3.993056。后者约60.85%，包含保持及正向后向迁移，不能全部称为减少遗忘。',
'- seed44是必须保留的反例：任务6阶段8下降25个百分点，最终下降10个百分点；其NBT变差3.333333。不存在“所有任务都被保护”的证据。','',
'## 4. 覆盖机制的直接证据','',
'对每个任务全部已导出窗口，复算其到实际保留集合的最近距离。使用原冻结的视觉/动作各0.5混合距离、原标定与原action尺度；没有重新选样。最大距离与Coreset原日志最终半径一致。下表相对降幅=(Random−Coreset)/Random。','',
'| seed | 任务 | 平均距离降幅 | 最大距离降幅 |','|---|---|---:|---:|']
for row in c['rows']:
 lines.append(f"| {row['seed']} | {row['task']} | {100*row['C_vs_R_mean_reduction_fraction']:.2f}% | {100*row['C_vs_R_max_reduction_fraction']:.2f}% |")
lines += ['',
'仅计算会被后续训练实际回放的任务6—8，九个seed×任务单元的相对降幅平均为：平均距离15.09%，最大距离55.82%。其含义是在同样缓存条目数下，原特征空间里远离缓存代表的窗口更少；它不是闭环成功率提升的直接证明。任务9列仅用于选择器诊断，不能解释本轮模型变化。',
'各任务两arm缓存约有一半成员不同，配额严格一致；没有通过增加缓存、训练步数或回放次数获得预算优势。全遍历会让这些不同成员反复参与后续更新。由此，较合理的机制是：保留更广的视觉—动作情形，改变后续回放的梯度内容，从而在部分种子/任务上改善旧任务和新任务的共同学习。这里“改善梯度内容”是机制推断，未直接测量梯度冲突或抓取行为。',
'反证边界：seed44任务6平均/最大距离改善17.12%/60.09%，覆盖改善甚至不弱于seed43，但后续黄油表现变差。因此距离覆盖不是充分条件，也不能用当前12个相关单元宣称覆盖与成功率有可靠因果/统计关系。该距离是选择器自身优化的代理指标，偏向它是预期现象；平均距离同时下降提供了额外描述，但仍不是独立行为质量指标。','',
'## 5. IFA与评估随机性的影响','',
'IFA日志：所有seed的阶段7均选择(6,7)，阶段9均无任务对。seed43/44两arm阶段8也均为空；只有seed42在阶段8出现Random选择(7,8)、Coreset为空。语言参考在各自缓存内的最大偏差全部为0。最大提升seed43并非通过“新增/删除IFA任务对”获得，但相同损失定义下实际表示和梯度仍会变化。seed42收益包含buffer引起的IFA选择变化，无法单独隔离为纯覆盖效应。','',
'本轮还只读加载了60份评估前RNG快照（不是模型checkpoint）。30个arm配对单元的Python/NumPy/CPU RNG均相同；CUDA RNG有10个不同。源码在每个阶段开始重置全局seed，但不是每episode独立generator；前面的10路组提前结束会消耗不同数量的动作随机数，影响之后的组/任务。','',
'因此：相同初态与seed保证了协议对齐，却不保证全程逐动作噪声相同。原生评估规则两组一致，这不使结果作废；但单个初态的成功→失败或失败→成功不能被当成排除了随机性的模型因果证据。30个单元中SS=404、SF=28、FS=71、FF=97（先Random后Coreset），只作描述；同初态跨阶段/种子重复，不能把600对当成600个独立训练重复做显著性检验。','',
'## 6. 可以与不可以写的结论','',
'可以写：在固定预算、state=True、LIBERO-OBJECT三个探索性seed的本轮协议下，Greedy Coreset相对配对Random取得平均AUC +6.56个百分点；特征—动作覆盖距离明显下降，主要闭环收益来自任务6后续表现及任务9最终学习表现，种子异质性明显。',
'不能写：已证明普遍显著优越、所有旧任务遗忘减少、几何覆盖是唯一原因、已精确恢复作者训练轨迹、状态模态开启解释了组间提升（两组都开启），或已证实抓取/夹爪切换行为更好（未做轨迹语义分析）。',
'保留协议缺口：官方预训练use_state元数据unknown；空间前端状态分支参数存在并加载通过，但无法据此恢复作者预训练开关。调度/warmup与最终checkpoint规则存在公开细节补全。RTX3090与论文A100硬件不同，cuDNN benchmark开启不保证跨硬件位级复现。三个seed还同时改变训练、导出、Random入库及评估RNG，不能分离这些方差来源。','',
'## 7. 全部成功率矩阵','']
for s,v in r['seeds'].items():
 for arm in ('random','coreset'):
  lines += [f'### seed{s} {arm}','', '| 阶段 | 任务6 | 任务7 | 任务8 | 任务9 |','|---|---:|---:|---:|---:|']
  for stage,row in zip(range(6,10),v['arms'][arm]['matrix']):lines.append('| '+str(stage)+' | '+' | '.join('—' if x is None else f'{x:.0f}%' for x in row)+' |')
  lines.append('')
lines += ['## 8. 复核命令与产物','',
'所有下列分析命令退出码均为0。服务器分析只用CPU读取既有特征、演示、缓存、RNG记录，不创建optimizer，不加载模型，不运行环境。','',
'```text',
'python -X utf8 execution/20261014_coreset_result_review/review.py',
'/data/mlr_ifa/env/bin/python -u /data/mlr_ifa/analyses/coreset_result_review_20261014/coverage_readonly.py /data/mlr_ifa/analyses/coreset_result_review_20261014',
'/data/mlr_ifa/env/bin/python /data/mlr_ifa/analyses/coreset_result_review_20261014/rng_readonly.py /data/mlr_ifa/analyses/coreset_result_review_20261014/rng_review.json',
'python -X utf8 execution/20261014_coreset_result_review/build_report.py','```','',
'RESULT_REVIEW.json：独立复算、全部单元SS/SF/FS/FF、指标贡献、IFA与配额/成本。coverage.json：12组真实载荷覆盖诊断；coverage_input_hashes.json：原输入前后校验清单；rng_review.json：逐单元RNG内容对比与输入哈希。分析代码及报告全部位于原run之外的新目录。']
(P/'REVIEW_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('PASS: report and coverage cross-validation; no original input writes')
