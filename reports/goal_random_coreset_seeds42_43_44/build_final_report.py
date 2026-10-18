"""Render all completed results from the independent audit; no experiment execution."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
R=Path(__file__).resolve().parent
d=json.loads((R/'FINAL_RECOMPUTED.json').read_text(encoding='utf-8'))
assert d['status']=='PASS'
e=R/'final_evidence'
q=e/'queue'
stamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
lines=['# LIBERO-GOAL 三 seed Random / Coreset＋IFA 最终报告','',f'状态：**COMPLETED**。本地独立验收时间：{stamp}。', '',
 '三 seed 的 Coreset AUC 均低于 Random，平均差 −3.020833 pp；本次 GOAL 结果没有复现 OBJECT 上的平均提升。负结果不影响协议验收通过。没有追加实验、挑选 checkpoint 或修改超参数。','',
 '共 2,400 个完整 epoch、456,600 次优化器更新、24 个 ep99 最终 checkpoint、1,200 条正式闭环记录、120 次十路 vector attempt，零技术异常。六组各 400 epoch / 76,100 step / 200 episodes。服务器独立进程实际加载全部 24 个 checkpoint，每个完整加载 100 个下游键；每 seed 两组 Task6 实际张量和初始化 RNG 一致。queue、wrapper、三个 seed 的 pipeline / supervisor 均退出 0，最终快照无残留实验进程。','',
 '## 指标（百分数；差值为百分点）','',
 '|Seed|方法|AUC↑|FWT↑|NBT↓|','|---|---|---:|---:|---:|']
for seed,a in d['seeds'].items():
 for arm in ('random','coreset'):
  m=a[arm]['metrics'];lines.append(f"|{seed}|{arm}|{m['AUC']['percent']:.6f}|{m['FWT']['percent']:.6f}|{m['NBT']['percent']:.6f}|")
 m=a['C_minus_R'];lines.append(f"|{seed}|C−R|{m['AUC']:+.6f}|{m['FWT']:+.6f}|{m['NBT']:+.6f}|")
lines+=['','三 seed 均值 ± 样本标准差（ddof=1，按 seed 统计，未把 episode 当独立训练重复）：','','|方法|AUC↑|FWT↑|NBT↓|','|---|---:|---:|---:|']
for group,m in d['stats'].items():
 lines.append('|'+group+'|'+'|'.join(f"{m[k]['mean']:.6f} ± {m[k]['sample_sd_ddof1']:.6f}" for k in ('AUC','FWT','NBT'))+'|')
lines+=['','定义：r(s,t) 为阶段 s 后任务 t 的 20 次成功率。AUC = ¼∑t mean(s≥t) r(s,t)；FWT = ¼∑t r(t,t)；NBT = ⅓∑t=6..8 mean(s>t)[r(t,t)−r(s,t)]。NBT 越小越好，可为负。精确有理数保存在 FINAL_RECOMPUTED.json。','',
 '## 全部成功率矩阵','', '行是训练完成阶段，列是被评估任务；每个有效单元固定 20 次。任务6：奶油奶酪放入碗；7：打开炉灶；8：碗放到盘子上；9：酒瓶放到架子上。','',
 '|Seed|方法|阶段|Task6|Task7|Task8|Task9|','|---|---|---|---:|---:|---:|---:|']
for seed,a in d['seeds'].items():
 for arm in ('random','coreset'):
  for stage,row in enumerate(a[arm]['matrix'],6):
   lines.append(f'|{seed}|{arm}|{stage}|'+ '|'.join('—' if x is None else f'{x:.0f}' for x in row)+'|')
lines+=['','## 差异集中在哪里','',
 'Task6 是三个 seed 的共同退步点：最终阶段 Coreset 比 Random 分别低 25、25、35 pp。Task7 六组所有已见阶段均为 100%；Task8、9 的收益方向不一致。seed43 的新任务学习表现（FWT）更高，却伴随更差的旧任务保持（NBT），所以 FWT 提升不能代替持续学习总体改善。','',
 '以下是各任务对总 AUC 差的精确分解，已包含总指标的 ¼ 权重，四列相加等于该 seed 的总 AUC 差；它描述差异位置，不证明机制。','',
 '|Seed|Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|']
for seed,a in d['task_auc_contributions_C_minus_R'].items():lines.append('|'+seed+'|'+'|'.join(f'{a[str(t)]:+.6f}' for t in range(6,10))+'|')
lines+=['', '不能仅凭这些结果认定具体操作片段缺失或距离设计错误；这需要另行授权的只读覆盖/轨迹审计。此次不追加训练或评估。','',
 '## 覆盖、配额和成本','',
 '每任务完整窗口数 997/889/904/1969，入库条目 499/445/452/985；两组预算相同。24 份 storage 清单的成员唯一、合同一致、阶段累计 buffer 哈希连续通过；服务器已逐项核对原 feature 与入库载荷哈希。Coreset 的 visual/action 距离设置、标定/参考 ID、动作尺度和完整覆盖半径曲线均在各组 storage/task*.json。Random 没有另算同一距离覆盖半径，不能声称已有同口径覆盖改善比较。未下载任何真实载荷。','',
 '|Seed|方法|训练调用分钟|epoch累计分钟|环境构建至关闭累计分钟|Coreset选样秒（四任务）|','|---|---|---:|---:|---:|---:|']
for seed,a in d['seeds'].items():
 for arm in ('random','coreset'):
  seconds=sum(json.loads((e/f'seed{seed}'/arm/'storage'/f'task{t}.json').read_text())['selection'].get('seconds',0) for t in range(6,10))
  v=a[arm];lines.append(f"|{seed}|{arm}|{v['training_seconds']/60:.2f}|{v['summed_epoch_seconds']/60:.2f}|{v['environment_seconds']/60:.2f}|{seconds:.3f}|")
lines+=['','成本来自真实计时，环境列含构建/关闭而不包括两 cell 间全部模型准备开销；选样秒已包含在训练调用中，不能再相加。完整开始/结束时间和命令见原 commands.log。','',
 '## 执行与验收证据','',
 '远端队列：`/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017`；产物：`/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{42,43,44}_20261017`。原始源码 commit `a721807f2f7fa7f9833dcc85c592d78e9accd00b`，执行补丁/冻结源码/配置/模态/官方权重与输入哈希随证据保存。','',
 '真实队列和六组执行命令及退出码原文如下（摘录 START/EXIT）：','', '```text']
for p in [q/'commands.log']+[e/f'seed{s}'/'commands.log' for s in (42,43,44)]:
 lines.append(str(p.relative_to(e)))
 lines.extend(p.read_text().splitlines())
lines+=['```','', '最终自动执行 `final_queue_audit.py`，FINAL_QUEUE_ACCEPTANCE=PASS，queue.exit=0，wrapper.exit=0。手动只读打包命令 `/data/mlr_ifa/env/bin/python /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/package_evidence.py` 退出0：941文件、24,799,387字节，下载后逐项核验。','',
 '本地及发布目录独立复算命令（退出0）：','', '```sh','python -X utf8 recompute_final.py','```','',
 '脚本仅用标准库：复算 1,200 条真实结果、120 次尝试、所有矩阵/精确指标/样本SD，检查 epoch/step/入库清单/配置/源码/加载凭证与包内 941 文件哈希，并检查输入前后未变。不导入训练框架、不加载权重、不调用环境。服务器已核验大载荷；本地依据下载的哈希和原验收凭证核查，不冒称本地重新加载权重。','',
 '首次预检和历史失败材料若在包内，按原文件保留；不把已解决的历史材料误写为正式失败，也不覆盖日志。此次正式三 seed 队列无需训练或评估恢复。','',
 '## 解释边界','',
 '- 三 seed 跨套件探索性比较；GOAL 三 seed 在本轮结果产生前指定。这里只作描述性结论，不宣称统计显著或普遍因果。','- 与论文明确参数对齐，但官方预训练 `cfg.policy.use_state` 缺失仍记 unknown；本轮实际三模态为 True，状态参数完整加载。','- warmup500、逐更新调度调用、ep99最终权重等公开细节缺口已明确补全，不能声称精确恢复作者训练轨迹；seed42来自官方脚本，正文未列具体seed。','- RTX3090 与论文 A100 不同；旧 GOAL state_v2 的 RNG/IFA/导出细节不完全等价，仅保留历史参考。','- 评估全局动作 RNG 每阶段重置，原 argmax GMM 分量后 Normal 采样；未逐轨迹隔离。初态、seed、配置对应，但两模型消耗随机数的顺序仍可能分岔。','- Coreset 改变 buffer 成员后 IFA 配对可能间接变化，结果属于完整 Coreset＋IFA 方案，不能隔离为单纯覆盖因果。','',
 '本轮完成后停止并取消 GOAL 进度定时检查。只发布代码、报告、配置、哈希、逐 episode 结果及技术尝试；不发布权重、演示、latent 或 buffer 载荷。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('FINAL_REPORT.md generated')
