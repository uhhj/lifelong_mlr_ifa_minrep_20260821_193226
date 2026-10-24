# CCBP GOAL Task8→Task9 四组有限验证

2026-10-24 20:55:40 CST实际观察：PARTIAL_TRAINING。四组从同一已验收Stage8检查点分叉，仅seed41；四组各100epoch33700更新，共134800更新，随后160正式回合。协议于首次部署前已发布c4c8fdf94e0b073d69c341d23f59c8e514673f46；尚无科学结果。

A ALIGN；B固定执行Normal行为蒸馏；C执行分量expert cost和路由；D CCBP。附件未指定参数已在PROTOCOL一次固定，禁止结果后扫描。D需Task8严格高于ABC且Task9不低于任一，才记支持；不满足不扩展。即使通过也不自动扩seed。额外前向不称等算力，expert cost不是闭环风险。

本轮自己的A/B/C/D隔离GPU准入全部实际PASS：A4次、BCD各3次optimizer调用共13次（含A原ALIGN对照1次），全部0环境。模型均100键、21633611参数。A更新与原ALIGN原生observe逐状态精确相等；四组序列化完整状态恢复后下一步精确重放。长度1–8/部署argmax Normal与路由空集合、全good及梯度检查通过。

教师Stage8固定、430合法Task8旧窗口，14前向、0更新。尺度translation0.46725568175315857、rotation0.10000000149011612、gripper1.0，只由固定训练回放拟合；epsilon.05/lambda.1/beta.1/margin.1不调整。

A9完整epoch3033正式更新；BCD正式0。每epoch3365样本、3218有效/147屏蔽、430旧Task8有效；当前GPU14538/controller14309正常。22冻结文件及父源码一致。A9轮平均9.180987秒纯计算，保存/准备额外；BCD额外前向未实测。暂估整段四组训练约90分钟量级，下一90分钟后检查；正式评估吞吐未测，不承诺全实验结束时间。当前不是闭环或最终接受。

系统普通空闲3619938304 bytes、data root空闲3458539520 bytes；新增2.3GiB界包含原子保存峰值并抵扣已分配，保留system1GiB/data2GiB，无删除/降预算/新付费。

第一次部署发送器在Python解析阶段发生Unicode传输错误，无服务器文件写入或优化器调用。ATTEMPT保留，外置发送器改ASCII转义后部署成功；未修改科学源码或重跑训练。冻结controller不自动恢复中断正式评估。

最终外置CPU和本地审计脚本已准备且仅AST通过，必须全队列自然退出后真实执行。届时复核4完整CP、全恢复/输入/teacher、400epoch134800步、160episode16vector8cell和逐窗口专家成本；完整分析Task8/9及D相对ABC的paired SF/FS。所有负结果、单seed限制和新配对噪声不同于旧global-RNG AUC协议的限制保留。

本地从已验收GOAL41包独立读取原Stage9 manifest，字节SHA4673c47a814b6c0bae97f071e108cff0615578ececc5c7c17ceee773060eea61与本次教师绑定相同；独立复算3365/3218/147/430并逐epoch核对通过，非只相信服务器累计计数。
