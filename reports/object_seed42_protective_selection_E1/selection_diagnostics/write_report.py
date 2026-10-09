import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;d=json.loads((P/'SELECTION_DIAGNOSTICS.json').read_text());a=d['summary']['all']
lines=['E1已有选样日志诊断：偏好是否落实、固定候选池是否形成短期重复集中',
'结论：分数确实形成中等程度、实际落地的检索偏好；优先分支存在明显的刷新周期内重复集中，主要由64个固定候选的支持集限制产生。没有出现概率退化到极少数候选的现象。不能将这种检索偏好解释为闭环保护有效，也不能由日志证明它导致主AUC下降。',
'',
'范围与只读保证',
'输入来自已验收seed42的本地真实日志，发布提交e963d8d0f3a4626365d33bdb5c83a60821fff115。读取400个epoch事件文件及catalog、合同、训练收据；406项输入与最终归档清单哈希匹配，全部输入前后哈希不变。',
'分析181次刷新、389个“阶段×旧任务×刷新周期”候选池、230710次实际优先赋值。任务6没有旧buffer、无选样；任务9在本次序列中只作为当前任务，不能伪造旧任务9选样结果。',
'只用Python/NumPy读取已有JSON日志；没有加载模型、环境执行、训练、rollout、随机模拟、修改参数、修改原run或push。',
'',
'1. 分数是否真正形成偏好',
f'最高分16/64候选：均匀抽样应占25%，冻结q预测{100*a["q_top16_mass"]:.3f}%，实际抽到{100*a["actual_top16_fraction"]:.3f}%。约为均匀水平的{a["actual_top16_fraction"]/.25:.2f}倍。',
f'最低分16/64候选：q预测{100*a["q_bottom16_mass"]:.3f}%，实际{100*a["actual_bottom16_fraction"]:.3f}%，低于均匀25%。',
f'候选池均匀平均分{a["uniform_mean_score"]:.6f}；按q抽样的预测平均分{a["predicted_selected_mean_score"]:.6f}；实际抽样平均分{a["actual_selected_mean_score"]:.6f}。实际较均匀提高{a["actual_selected_mean_score"]-a["uniform_mean_score"]:.6f}。各池以真实抽样次数加权。',
f'q的有效样本数ESS=1/Σq²，抽样次数加权平均{a["q_ess_mean"]:.2f}/64；各池范围49.61—62.43。归一化熵均值{a["q_entropy_mean"]:.4f}（均匀为1）。最高分组概率在所有池中均大于25%，范围29.82%—46.20%。',
f'最高/最低候选概率比平均{a["q_max_min_mean"]:.2f}，中位数5.10；这不是只选最高分。实际高分份额与q预测仅差{100*(a["actual_top16_fraction"]-a["q_top16_mass"]):.3f}个百分点。',
'这排除了“计算分数却实际仍均匀抽样”这一解释。偏好针对当前日志中的头部梯度代理分数；日志没有证明高分样本的完整网络梯度或闭环价值更好。',
'',
'2. 短期重复的定义与结果',
'每个阶段/旧任务/刷新周期重新计数：重复率=(抽样次数−不同文件ID数)/抽样次数。每个ID在该窗口第一次出现不算重复。相邻刷新窗口不合并，任务也不合并。该指标只覆盖已记录的优先赋值分支。',
'理论参照均使用相同候选数/旧任务buffer大小和每池实测抽样次数，假设有放回独立抽样；它们不是新跑的实验或实际E0轨迹。原E0是拼接数据集shuffle，不能把全buffer有放回参照直接标成E0。',
'刷新后窗口    实际优先分支重复率    同64池均匀(理论)    全任务buffer均匀有放回(理论)']
for h in ('50','100','500'):
 x=d['horizons'][h];lines.append(f'前{h}step       {100*x["repeat_fraction"]:.2f}%              {100*x["repeat_fraction_expected_uniform_pool"]:.2f}%               {100*x["repeat_fraction_expected_full_buffer"]:.2f}%')
lines += ['500step表仅含完整500step周期；末尾不足500的周期不冒充完整周期。50/100step分析按该窗口过滤事件。',
f'全部周期（含末尾短周期）：实际重复率{100*a["repeat_fraction"]:.3f}%，已知q预测{100*a["repeat_fraction_expected_weighted_pool"]:.3f}%，同64池均匀预测{100*a["repeat_fraction_expected_uniform_pool"]:.3f}%。每池平均{a["draws"]/a["pools"]:.1f}次抽样，平均{a["unique_pool_id_pairs"]/a["pools"]:.2f}个不同候选，池内多数候选都被用到，但同一批ID被多次使用。',
'',
'3. 区分固定池与分数加权的贡献',
'从同一个刷新周期、同一个旧任务中随机取两个不同抽样位置，检查文件ID相同的概率；比简单长窗口重复率更能区分权重偏斜。各池按可比较位置对数加权。',
f'全buffer均匀有放回(理论)：{100*a["full_buffer_pair_collision"]:.4f}%。',
f'同64候选池均匀(理论)：{100*a["uniform_pool_pair_collision"]:.4f}%。',
f'同64池按真实q(理论)：{100*a["q_pair_collision"]:.4f}%；实际：{100*a["actual_pair_collision"]:.4f}%。',
f'固定64池相对全buffer理论参照，将同ID碰撞提高约{a["pool_restriction_collision_factor"]:.2f}倍；分数加权在该基础上再提高约{100*(a["score_weighting_collision_factor"]-1):.1f}%。实际总倍数约{a["actual_collision_factor_vs_full_buffer"]:.2f}。这是概率结构分解，不是闭环性能因果分解。',
f'相邻两次同任务优先抽样立即抽回同ID的实际比例仅{100*a["observed_adjacent_repeat"]:.3f}%（q预测{100*a["expected_adjacent_repeat"]:.3f}%）。主要现象是500步内反复使用一个小池，而不是长串连续只抽一个ID。',
f'各池单ID最多赋值次数的中位数21、最大40；重复再访中{100*a["revisit_within10_fraction"]:.1f}%发生在10个optimizer step内、{100*a["revisit_within50_fraction"]:.1f}%在50步内。分母为再次出现的事件，不是全部抽样。',
'',
'4. 全部旧任务分组（不省略结果）',
'阶段/旧任务   优先赋值数   高分前25%实抽份额   ESS/64   周期内重复率']
for key,x in d['summary'].items():
 if key=='all':continue
 lines.append(f'{key:<15} {x["draws"]:>6}      {100*x["actual_top16_fraction"]:>6.2f}%         {x["q_ess_mean"]:>5.2f}    {100*x["repeat_fraction"]:>6.2f}%')
lines += ['',
'5. 刷新是否一直留着同一批候选',
f'相邻刷新之间，同任务两个64候选池平均重叠{d["overlap_mean"]:.3f}个ID；独立均匀重抽候选集的参照为{d["overlap_expected_mean"]:.3f}个。没有看到候选池在刷新后仍长期固守少数固定ID的整体迹象。',
'该观察与先前全阶段优先分支覆盖767/781、1513/1521、2299/2301个合法旧buffer ID一致：问题集中在短期窗口，不是整段训练完全不覆盖其余缓存。',
'',
'6. 边界与不能作出的推断',
'原DataLoader保留分支的具体ID没有完整写进选样事件，因此不能直接计算整个回放流的唯一覆盖、所有旧样本之间的实际碰撞，也不能知道优先赋值是否恰好等于被替换的原ID。以上“重复”仅指优先分支内部再次选到同ID。',
f'若仅用50%全buffer均匀+50%优先q的概率模型，固定候选池预计承载约{100*a["mixture_model_candidate_mass"]:.2f}%的旧样本检索；它不是完整训练流的实测比例。该混合模型有效支持数约{a["mixture_model_effective_size"]:.1f}，不能用优先分支13.51倍碰撞直接形容全部回放。',
'池内没有每step重评分；无法判断分数在500step内是否陈旧、参考梯度方向是否已变。日志只保存合成分数，没有分别保存历史对齐项和新任务冲突项，不能分拆二者作用。',
'文件ID是窗口身份，未必是独立演示或不重叠时间段；不同文件仍可能语义相近，不把文件级集中当作完整轨迹级多样性。',
'已知主AUC下降0.417个百分点只能作为实验结果背景，不能凭这些日志把下降归因给固定候选池，或把代理分数上升当作保护有效。没有自动改温度、刷新间隔、候选数、采样比例或启动下一组。',
'',
'复算定义',
'q=softmax(score)，ESS=1/Σq_i²，归一化熵=−Σq_i log(q_i)/log(64)。',
'n次抽样的预期不同ID数=Σ_i[1−(1−q_i)^n]；均匀M池时=M[1−(1−1/M)^n]。',
'实际同ID位置对率=Σ_i c_i(c_i−1)/(n(n−1))；其理论期望=Σq_i²。',
'50/50混合检索的同ID概率=0.75/M+0.25Σq_i²。全buffer有放回参照仅作解析比较。',
'',
'实际命令：python -X utf8 execution/20261009_E1_selection_diagnostics/analyze_selection_logs.py；退出码0。',
'产物：SELECTION_DIAGNOSTICS.json（分组/短窗口/批内碰撞/边界）、per_pool.csv（389行）、refresh_overlap.csv（383行）、INPUT_HASHES.json（输入前后及归档验证）、分析脚本。',
'所有产物位于本目录，原实验与Git发布分支均保持不变。']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.is_file() and f.name!='OUTPUT_MANIFEST.json'}
(P/'OUTPUT_MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
print('Report saved; no input changes.')
