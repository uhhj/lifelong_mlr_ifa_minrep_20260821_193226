"""Independently check lightweight temporal evidence and report the failed causal gate."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
P=R/'caet_temporal_audit_20261021_191704'
raw=(P/'EVIDENCE.json').read_bytes();d=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==json.loads((P/'RECEIPT.json').read_text())['sha256']
assert d['status']=='AUDIT_EXECUTION_PASS' and not d['cuda_initialized']
assert all(d[k]==0 for k in ('optimizer_calls','environment_calls','server_writes','model_forward_calls'))
total=valid=checks=demo_instances=0
for seed,s in d['seeds'].items():
 assert s['feature_symlink_resolves_to_parent']
 for name,source in s['sources'].items():
  assert hashlib.sha256(source['text'].encode()).hexdigest()==source['sha256']
 model=s['sources']['caet_exp/model.py']['text'];fit=s['sources']['caet_exp/fit.py']['text']
 assert 'feature[:,:,1:,:].mean(1)' in model and 'torch.cat((x,action),-1)' in model
 assert 'r[0]["actions"][0,-1]' in fit and 'prediction,embedding=model(x,a)' in fit
 assert 'load_next_obs=False' in s['sources']['libero/lifelong/datasets.py']['text']
 assert 'data["actions"].cpu()' in s['sources']['libero/lifelong/models/base_policy.py']['text']
 assert "data['obs'][key]" in s['sources']['libero/lifelong/models/complete_policy.py']['text']
 for task,v in s['tasks'].items():
  assert v['windows']==v['valid_effect_windows']+v['invalid_terminal_windows']+v['ambiguous_windows']
  assert v['valid_effect_windows']==v['valid_windows_with_last_input_post_target_action']>0
  assert v['cached_action_windows_and_feature_hashes']=='PASS'
  assert all(x['post_action_joint_exact'] and x['post_action_gripper_exact'] and x['different_from_pre_rows']>0 for x in v['demos'].values())
  assert all(x['same_index_obs_actions'] and not x['load_next_obs'] for x in v['native_dataset_checks'])
  total+=v['windows'];valid+=v['valid_effect_windows'];checks+=len(v['native_dataset_checks']);demo_instances+=len(v['demos'])
out=dict(status='LOCAL_EVIDENCE_VERIFICATION_PASS',causal_temporal_gate='FAIL_POST_ACTION_INPUT',
    seed_task_windows=total,valid_effect_windows_including_post_action_input=valid,
    native_sequence_checks=checks,seed_demo_instances=demo_instances,
    independently_reencoded_features=False,scope='Physical-file alignment + actual dataset indexing + frozen encoder/consumer provenance + all cache hashes/actions; no new model forward.',
    training_calls=0,environment_calls=0,source_sha256=hashlib.sha256(raw).hexdigest())
(P/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
report='''# CAET OBJECT 零训练时间信息审计：发现动作后输入

**结论：未通过“效果预测器不能看到目标动作执行后观测”的要求。现有实现确实包含动作后的观测信息，不能称为严格的动作前效果预测。** 审计程序执行/证据核验PASS，与该科学要求FAIL是不同的两件事。

检查三seed41/42/43、Task6–9全部缓存（每seed5921个窗口），累计17763个缓存哈希和动作窗口；每seed5601个有效效果窗口，合计16803个，其最后一帧输入都对应目标动作后的观测。三seed复用同一批原始演示，这些数量不是独立统计样本。

## 证据链

1. 实际HDF5中，对每个示范完整序列精确核验：`obs/joint_states[t] == states[t+1, joint_qpos]`、`obs/gripper_states[t] == states[t+1, gripper_qpos]`。这在三seed×四任务×十示范的120个seed-demo绑定中全部成立；并非只是注释或对标准数据格式的假定。同索引的动作前state与obs确有大量不同。
2. 实际安装的SequenceDataset使用相同索引切取obs和actions，`seq_length=8`、`frame_stack=1`、`load_next_obs=False`。用每个demo的一条原生窗口（120次）核验，动作及关节/夹爪观测均按同索引返回；观测比较严格遵循ObsUtils原有float32转换，不放宽误差阈值。
3. 对全部缓存验证原文件SHA与窗口动作匹配、末端target身份。CAET特征目录直接链接到对应RA父特征，FEATURES清单和父合同绑定一致；冻结编码器将同索引关节/夹爪编码到state token，将同索引图像编码到视觉token，未对动作做时间前移。
4. 冻结EffectModel输入为8帧的两个视觉token和一个state token的时间平均，即 `feature[:,:,1:,:].mean(1)`，然后拼接该窗口末端专家动作 `actions[0,-1]`。最后一帧没有被剔除，也没有因果mask；平均操作不会消除动作后信息。
5. 标签是目标物体的 `states[t] → states[t+1]` 平移与旋转变化。因此输入含有该转移结束后的观测。未来状态仅用于构造训练标签本来是合法的，问题在于它还经缓存观测进入预测器输入。

示意：`[obs[t-7], …, obs[t]] + action[t] → effect(states[t],states[t+1])`，但`obs[t]`已经是`action[t]`执行后的观测。这不是显式读取`next_obs`字段才会出现的问题。**至少机器人关节/夹爪后状态泄漏已由原始数值精确证明**；视觉路径按同一obs索引进入编码器，但本轮没有重新渲染图像或重算视觉特征，不能冒称做过像素级物理时序复验。

## 如何解释已有CAET结果

- 三seed闭环分数及其独立复算仍是实际运行结果，不能删除或重写。既有工程验收没有覆盖这项因果时间要求，应补充本次失败结论。
- 效果预测器可能利用“已经发生的结果”降低预测误差；非零动作梯度、跨演示留出误差和正对数量都不足以排除这种捷径。因此此前“学得可迁移的动作效果规律”的解释证据不足。
- 在策略辅助训练中，同一后状态特征与新采样的候选动作组合，却仍匹配原专家动作效果；这可能产生缺乏因果依据的梯度。但本次没有消融，不能断言它就是CAET所有收益或退化的唯一原因。
- RA与CAET共同继承原缓存配对时序，这不是CAET才改变了基线数据；但CAET把它用于“预测动作后果”时引入额外解释问题。不能用共同基线来证明因果预测输入合法，也不能因此直接宣告所有历史控制分数无效。

## 处置边界

保留原冻结源、权重和报告，仅补充审计。不训练、不重评、不修正既有标签或重导特征。若以后修复，应另立版本：输入只使用目标动作执行前确实可获得的观测，目标仍为后续效果，并重新核验窗口/状态/视觉时间绑定和共同基线可比性；不能只将标签索引挪一格后宣称协议不变。本轮没有授权或执行这项新实验。

原先只读结果分析中的AUC +1.527778pp保留为描述性现象，其机制解读必须附带本审计限制。CAET-GOAL不应仅解决正对数量后就启动，时间信息问题也必须先处理。

## 审计执行说明

全程CPU、0模型forward、0优化器、0训练、0环境、0服务器写入，无权重/状态/动作张量下载。仅下载源码和元数据。三次外置只读尝试分别因AMBIGUOUS窗口过滤、ObsUtils初始化、原生float32转换比较处理不完整而退出，失败证据保留；只修复审计脚本，没有改实验。最终全量核查与本地独立证据复核通过。未重新编码特征，结论依赖实际缓存身份、冻结导出路径与原生索引的联合证据，边界明确保留。
'''
(P/'REPORT.md').write_text(report,encoding='utf8')
print(json.dumps(out,indent=2))
