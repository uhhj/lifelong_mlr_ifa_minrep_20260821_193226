"""Write the decision report from verified zero-training metadata."""
import argparse,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();p=a.directory
 d=json.loads((p/'LOCAL_AUDIT.json').read_text());assert d['status']=='LOCAL_INDEPENDENT_PASS'
 lines=['# CAET GOAL 零训练正对与真实组批核查','',
 '**结论：不支持直接采用“Task7关闭、Task8/9全部启用”。Task7应回退RA+IFA；Task8支持过稀，不建议当前版本启用；Task9有值得小规模验证的迁移支持，但尚未证明闭环增益。**','',
 '此为已有资产CPU分析：三seed41/42/43，全部4759窗口/seed真实动作窗口身份与物理状态时序重新验证，按每阶段实际RA记忆重建候选。没有实例化策略或优化器，没有训练、梯度更新、GPU计算、模拟器或额外闭环。运行中的COVR及原冻结队列没有改变。','',
 '## 当前任务与旧任务拆分','',
 '正对采用冻结规则：不同任务、双方非静止（1mm或1度）、训练集逐维标准化后距离≤0.5。以下是**无向窗口对**，原报告有向掩码计数为其两倍。演示ID以任务为命名空间，跨任务同名demo不当成同一演示。完整演示留出不用于拟合或尺度统计。','',
 '|Seed|阶段|当前—旧正对|旧—旧正对|当前非静止窗口被覆盖|全局当前演示覆盖|有效batch/总batch|batch中实际当前演示覆盖|',
 '|---|---|---:|---:|---:|---:|---:|---:|']
 for seed,sd in d['summary'].items():
  for stage,x in sd.items():
   lines.append(f"|{seed}|{stage}|{x['current_old_unordered']}|{x['old_old_unordered']}|{x['current_windows_supported']}/{x['current_moving_windows']} ({x['current_moving_coverage_pct']:.2f}%)|{x['current_demos_supported']}/9|{x['current_old_effective_batches']}/{x['effect_batches']} ({x['current_old_effective_batch_pct']:.2f}%)|{x['used_current_demos']}/9|")
 lines+=['','## 配对究竟来自哪里','',
 'Task7所有seed均为0，既不是漏读物理效果，也不是仅仅错过batch。当前训练集合中的222个非静止窗口没有符合规则的旧任务邻居。非静止跨任务距离与0.5边界最近也相差至少0.35986，当前零正对结果不贴近浮点边界。','',
 'Task8的12/12/18个正对全部来自Task6↔Task8，没有Task7↔Task8。全局当前支持只有7/7/9个独立监督时刻，占非静止窗口1.92%/1.92%/2.47%。虽然分散在5/9条当前演示与6/6/7条旧演示，不是单一demo偶然重叠，但每条支持很薄。','',
 'Task9当前—旧配对234/308/275，旧—旧只有3/1/0，不存在旧—旧配对虚增主结论的问题。分任务为：','',
 '|Seed|Task6↔9|Task7↔9|Task8↔9|Task6↔8（旧—旧）|',
 '|---|---:|---:|---:|---:|']
 for seed,sd in d['summary'].items():
  x=sd['9']['task_pair_counts'];lines.append(f"|{seed}|{x.get('6-9',0)}|{x.get('7-9',0)}|{x.get('8-9',0)}|{x.get('6-8',0)}|")
 lines+=['','约76%–78%的Task9当前—旧配对来自Task7；这说明旋钮效果并非对所有后续任务都不可匹配。匹配仍依赖当前六维物理表示与阶段标准化，不能从数字相近直接推导同一控制规律可复用。Task9全局覆盖9/9条当前演示、23/24/23条旧演示，最多单一演示对占4.55%–5.56%，没有由一对演示垄断；但只覆盖44–52个当前监督时刻，仍仅占非静止窗口6.02%–7.11%。','',
 '## 原batch中是否有效','',
 '核查的是效果模型fit.py的batch64、固定10轮seed顺序；原生文件排序→buffer+current拼接→Python seed打乱→有效标签过滤→整demo留出→训练索引torch.randperm，与冻结实现一致。CPU逐batch重算pair_masks，并用无梯度零embedding调用原transfer_loss核对计数。必须同一anchor同时有正对和负对，才计为有效；不是只看同批出现一对相似样本。主策略batch10直接使用冻结效果模型预测误差，不在该batch执行正对对比，因此没有伪造batch10对比覆盖指标。','',
 'Task8仅7/2/8个batch有当前—旧有效信号，对应14/4/18次有向正对使用。seed42前8轮完全没有当前—旧对比，仅最后两轮各出现一次无向配对。真正作为当前anchor参与损失的样本仅5/2/5个，覆盖4/2/4条演示。因此“全局非零就启用”很容易得到名义上的迁移模块、实际近似物理回归的训练。','',
 'Task9有33/48/46个有效batch，对应80/162/130次有向使用，每个seed的10轮均有信号。实际当前anchor为15/20/19个，当前演示覆盖6/8/9条。全局覆盖比Task8宽，真实batch也确实能利用；但89.6%–92.8%的batch没有当前—旧对比，不能宣称充分、稳定或已经学到通用效果表示。','',
 '## 留出与泛化边界','',
 '以训练集尺度，独立当前留出demo的非静止窗口能匹配旧训练标签：Task7为0/24；Task8为3/37、3/37、4/37；Task9为3/61、4/61、4/61。支持不是严格只存在构造训练演示，但留出覆盖仍很低。这只是标签空间重叠，不是已学习模型的跨演示预测准确率，更不是闭环泛化证明。','',
 'CPU重放不是运行GPU训练。全局正对总数与先前独立可行性证据逐seed逐stage完全一致；最近0.5阈值距离最小约4.96e-5，仍需在未来真正准入时保留原GPU数值检查。本轮没有调整阈值、归一化、采样、记忆或增加预算。','',
 '## 决策建议（未执行）','',
 '1. 不采用“存在任意正对即可启用”的条件。Task8满足这个弱条件却只有0.8%–3.2%的batch有迁移信号。','2. Task7完整关闭CAET辅助策略损失，保留RA+IFA；Task8当前证据也倾向关闭。','3. Task9值得作为新命名协议的有限探索候选：具有跨演示的当前—旧支持且每轮真实组批可用。但启用前仍需明确效果预测的整演示留出质量和动作敏感性验收，不能用本轮零训练分析冒充已通过。若随后获授权，可先小规模验证仅Task9启用；不应立刻启动三seed整套实验。','4. 任何条件启用都应事前固定可解释的支持和模型质量规则，对所有阶段一致应用；本轮数据是设计依据，不能将事后挑选Task9写成预注册结论。不要通过增大阈值、重采正对或改batch来“修复”当前结果，这些均是另一个算法。','',
 '当前CAET-GOAL原冻结版本继续保持适用性阻塞。本轮仅交付建议，未改变自动队列或启动条件CAET。','',
 '## 证据与工程记录','',
 'PAIR_EVIDENCE.json仅含样本身份、演示/时刻元数据、pair边、固定batch索引和聚合统计，不含原动作、权重、特征、状态或RNG二进制；RECEIPT绑定哈希。audit_goal_pairs_local.py独立从边与batch重算全部当前—旧/旧—旧计数和覆盖，并检查每轮训练索引恰好覆盖一次、留出排除及实际计数。物理标签由服务器只读核验，不下载原数据。','',
 '首次本地发起的只读脚本在AST加载函数签名时缺少FileTrajectoryDataset类型声明而退出；发生在任何实验资产读取前，0优化器/0环境。保留goal_pair_audit_20261021_185431/ATTEMPT.json；只将声明移到函数加载之前，并补充batch索引证据。成功尝试为本目录，未触碰科学训练源或恢复实验。']
 (p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print(str(p/'REPORT.md'))
if __name__=='__main__':main()
