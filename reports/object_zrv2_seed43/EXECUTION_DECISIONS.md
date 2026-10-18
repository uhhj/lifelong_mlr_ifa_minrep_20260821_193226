# ZR-v2 seed43 执行登记

日期：2026-10-18。状态：准备与真实输入准入，尚未校准或闭环。

已完整读取附件 PROTOCOL.md、spec.yaml、reference_core.py、compare_metrics.py、test_reference.py 及用户粘贴方案。独立 PROTOCOL (3).md 与包内 PROTOCOL.md 的 SHA256 相同：71e40e2cf6349d5476e4bae5b163f424ed88eb5c014f4096be90ba29e433878f。

父运行固定为 Seed43_Random_Coreset_repeat_20261016_v1/coreset，实际四阶段模型及记忆必须与其 200 条记录绑定；首次 seed43 与 ZR-v1 不能代替。

用户最新明确授权覆盖附件“通过后仅讨论、不自动扩展”的措辞：seed43 工程验收和四项精确指标门槛全部通过后才扩展其他已有研究 seed（40、41、42、44），其他数值/算法不变。各 seed 必须另行绑定自身对应 Coreset+IFA 父模型和完整来源，不能混用 seed43 权重、缓存、记忆或评估记录。

用户已明确允许本轮及通过门槛后其他 seed 的新产物放系统盘。工作区 /root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018；产物 /root/mlr_ifa/experiments/ZRv2_OBJECT_seed43_20261018。旧资产只读。

服务器准入观察：RTX3090 空闲；/data 可用约975 MiB；/root 可用约23 GiB。repeat1 历史为 RTX4090，必须披露跨硬件历史对照与原全局动作随机流限制。

reference 合成软件测试不计为真实 GPU 集成验收。真实准入脚本仅核验父来源和语言门，无 optimizer、无环境。若冻结门阈值不通过，停止正式校准，不改阈值、不硬归零、不用任务ID替代推理。

预算：最多4次隔离真实 optimizer 调用、0环境步骤；正式40校准epoch/10550步/105240标签曝光、4个校准epoch9快照、200闭环。原公共训练与父评估不重跑。发布只包含代码、协议、诊断数值和哈希，不含任何模型、latent、buffer载荷或语言向量数组。
