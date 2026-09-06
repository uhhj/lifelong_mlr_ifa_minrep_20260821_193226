# 配对rollout与16案例核查：最终报告

## Verdict

**PAIRED_ROLLOUT_DIAGNOSIS_COMPLETE_WITH_LOCALIZED_FAILURE_MODES**

180个配对诊断trial及20个记录一致性trial已完成；随后逐例核查指定的16个方向性
案例，未新增rollout。主要行为差异已定位到**拿起并保持目标、再转入运输**这一
衔接。尚未确定训练层面的因果机制，也未验证任何新损失或梯度门控。

## 来源与边界

- 起点：`e65a64bfa3e7d7c829ff3ec840feeaa08f52a9f9`。
- 分支：`codex/paired-rollout-failure-diagnosis`。
- 服务器：dm-MS-7D70，项目`/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep`。
- 使用固定MLR/旧取样IFA/修正IFA模型、ep99、单张GPU0 RTX4090；不训练、不改权重。
- 原生`test.py`、`metric.py`、动作采样与success条件未修改，只增加被动vector环境
  记录包装；全程20trials、10env×2轮、5步settling、最多600policy steps。
- initial-state数组及settling后sim state跨模型逐项相同。记录轻量/完整版本的
  20个控制trial中，所有action、done、step及initial state完全一致。
- 每个目标task独立调用evaluator，随机数调用顺序不同于历史完整矩阵；以下成功率
  是独立诊断结果，不覆盖原正式指标，也不声称每个跨模型trial的动作噪声完全相同。

## 完整诊断成绩

每格20trial；百分比。

| 对照点 | MLR | 旧IFA | 修正IFA | 历史对应MLR/旧IFA/修正IFA |
| --- | ---: | ---: | ---: | --- |
| seed43 / stage7 / task7 | 100 | 95 | 95 | 100 / 100 / 85 |
| seed42 / stage9 / task7 | 90 | 100 | 75 | 85 / 100 / 65 |
| seed43 / stage9 / task9 | 85 | 40 | 85 | 80 / 25 / 80 |

task7首学差距缩小，说明原差距有评估调用/RNG敏感性；不能将其全部归因于训练机制。
task7后续保留差距和task9修正收益在这次调用下仍存在。

| 对照点 / 修正IFA相对 | 双方成功 | 仅修正成功 | 仅对照成功 | 双方失败 |
| --- | ---: | ---: | ---: | ---: |
| seed43 task7首学 / MLR | 19 | 0 | 1 | 0 |
| seed43 task7首学 / 旧IFA | 18 | 1 | 1 | 0 |
| seed42 task7保留 / MLR | 13 | 2 | 5 | 0 |
| seed42 task7保留 / 旧IFA | 15 | 0 | 5 | 0 |
| seed43 task9 / MLR | 15 | 2 | 2 | 1 |
| seed43 task9 / 旧IFA | 6 | 11 | 2 | 1 |

## task7的5个独有失败：3次未抬起，2次抬起后在途中落下

指定trial：`5,8,16,17,18`，对应MLR全部成功。

- trial5/8/17：修正版最大目标升高仅`0.60/0.46/0.67cm`，目标未进入持续携带状态。
  最近末端—目标距离为4.35–5.58cm；可见接近目标，但不能据此证明形成有效手指接触。
- trial16：step79越过3cm，step99达到17.60cm峰值，step103已低于3cm；此时末端
  高度仍为29.21cm，物体高度6.46cm，末端—物体距离23.66cm。
- trial18：step69越过3cm，step102达到20.33cm峰值，step110已低于3cm；此时末端
  高度32.06cm，物体高度9.34cm，末端—物体距离23.38cm。
- 两个回落案例始终未进入篮子中心XY10cm邻域，最近分别29.79cm和14.98cm。
  图像显示物体落在途中，而机械臂后续仍到篮子侧。这不是单纯“篮子内放歪”的表象。
- trial17可见重复接近目标，说明不能把所有失败描述为完全不响应状态。

## task9的11个独有成功：主要跨过了旧IFA未能完成的抬升环节

指定trial：`4,6,9,11,12,13,14,16,17,18,19`。

旧IFA在这11例均未超过3cm抬升阈值，最大升高0.47–0.96cm；其最近末端—目标距离
为3.49–5.62cm。图像中可见目标遗留/倾倒，机械臂后期转向篮子，而非成功携物运输。
修正版在这些配对初态下达到18.64–22.33cm抬升并最终官方成功，首次超过3cm在
step58–105，成功在step119–364。

这支持“此组收益主要发生在取得并保持目标的环节”的描述，而不支持“仅改善了
篮内最终放置”。但trial6/13/18在篮子处仍有较长过程（成功步364/257/329），
也不能宣称修正版放置阶段毫无问题。另有2个反方向trial，收益不是逐例单调的。

## 共同的可观察现象与尚未证明的机制

在上述16例的失败侧，目标均没有进入篮子XY10cm邻域，但末端后来进入了该邻域。
5个task7失败首次发生这种空间分离的步数为177/126/125/280/112；11个task9旧IFA
失败为124–223步。此几何补充是在视觉核查后提出的探索性描述，不是预注册分类器。
成功轨迹也可能短暂跨过这两个10cm边界，不能只凭一次该事件预测失败。

**工作假设：值得检查策略是否可靠识别“目标已拿起/仍保持”的真实状态，并据此
调整后续运输行为。** 它比“所有负梯度余弦都应被抑制”更贴近现有行为证据。
不过，观察到失败后仍去篮子不等于已证明模型没有这种状态表示；也可能是夹持姿态、
动作时机、动作噪声或分布外状态恢复造成。本轮没有记录接触对/接触力，没有隔离
改变某个更新或输入的因果干预，所以不能归因为IFA某个具体梯度或某一latent维度。

这16例是按结果方向指定的条件样本，不能拿其频率代表全部初态、seed或任务。
全部180个trial及反方向结果都在表中；不做显著性检验，不推翻原正式指标。

## 资产与复核

- `SUMMARY.json`、`per_trial.csv`、`paired_trials.csv`：全部正式诊断记录。
- `INSTRUMENTATION_PARITY.json`：20trial记录无扰动检查。
- `focused_events.csv`：16对共32条全步事件摘要。
- `FOCUSED_VISUAL_REVIEW.md`：逐例人工视觉结论，16例全部看过。
- 两个example manifest记录预览路径；原始npz、sim state与图片留在忽略的`runs/`，
  不提交到Git。图像每50步一帧，精确事件步数来自每步状态。

实际执行成功的命令（仓库根目录，使用既有Python环境）：

```bash
bash scripts/run_paired_rollout_diagnosis.sh
bash scripts/summarize_paired_rollout_after_queue.sh
/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python scripts/inspect_paired_rollout_cases.py
```

前两条以nohup在服务器运行，分别于2026-09-06 14:45:37、14:45:40结束。
本次16例核查只处理已存记录，没有再跑试次。所有新增内容为scripts及本报告目录；
`libero/`、训练配置、checkpoint、features和buffer未修改。

## 下一步

先在已有轨迹上验证“拿起/保持/丢失”的行为事件标签是否可靠，并区分动作几何
问题与失败后继续运输的状态响应问题；在这些标签有证据支持前，不设计新IFA损失
或门控。本轮不启动该后续实验。
