# PiPER 夹爪目标采纳与驱动响应核查

设备为现场记录的 S-V1.9-1。核查使用 rlmi 上位机实际安装的 `piper_sdk 0.7.0a1`、11 月 22 日三段独立 CAN 记录，以及 11 月 25 日最近两轮的执行器记录。全程没有向设备发送 CAN 帧。

## 控制器采纳目标的可观测范围

SDK 的 `GetArmGripperCtrl_V3()` 返回收到的 0x159 控制消息缓存。更新函数只处理控制消息类型，所读值来自命令载荷。0x2A8 返回实际开度、力度反馈和状态。两者都没有“当前已采纳的目标”、指令序号或接受／拒绝字段。0x476 的设置指令应答适用于 0x4xx 设置类消息；独立监听当时也未订阅 0x476。0x47E 是示教倍率、行程和摩擦等配置的反馈。公开的 0x251–0x256、0x261–0x266 对应六个关节驱动器，无法用来读取夹爪电机实际电流。已将七个安装文件与松灵官方代码逐行比对。[读控制消息](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/interface/piper_interface_v3.py#L340) · [缓存更新](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/interface/piper_interface_v3.py#L498) · [设置应答](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_set_instruction_response.py) · [官方 CAN ID 表](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v3/can_id.py)

停滞窗口的 190 条 0x159 都带 SocketCAN 本机发送标记，相同 ID 没有记录到外部来源的目标。宽度使能码始终为 1，力度指令原始值均为 1000，置零字节为 0。1,370 帧设备反馈均为开度模式、状态 0x40，即驱动报告已使能，未置欠压、过流或传感器故障位。机械臂反馈保持 CAN / MOVE J 模式，六轴实际角度均有变化，窗口内未记录到 CAN 错误帧或 STOP。以上能够排除所记录通道上的整机停机现象，单个夹爪目标是否在内部被采纳仍然未知。

## 发令之后的力度反馈与开度

统一采用 0x2A8 第 4–5 字节的有符号原始计数。已安装 SDK 将每计数解释为 0.001 N·m；松灵 pyAgxArm 的 AGX 夹爪定义将同一字段标成 mN。两套实现均按有符号整数解码，数值变化与发生时刻可直接比较。S-V1.9-1 这套硬件上的物理标定、采样来源以及电流换算仍需厂商确认。[安装版本的字段定义](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v3/feedback/arm_feedback_gripper.py) · [pyAgxArm 的字段定义](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/pyAgxArm/protocols/can_protocol/msgs/effector/agx_gripper/default/feedback/arm_feedback_gripper.py)

| 11 月 22 日，北京时间 | 首条 0x159 后首次反馈 ≤ −200 计数 | 首次实测闭合 ≥0.3 mm | 力度反馈最负值 | 整段反馈开度范围 |
|---|---:|---:|---:|---|
| 13:55:50 | 143.0 ms | 153.0 ms | −506 | 3.4–9.8 mm |
| 14:16:09 | 109.5 ms | 119.3 ms | −524 | 5.0–9.9 mm |
| 14:33:46 | 未出现 | 未出现 | −131 | 始终 9.8 mm |

在 14:33 停滞段，指令前 300 ms 的力度反馈中位数为 −10.5 计数，之后 6.85 秒为 −11。最负的一帧是 −131，整段只有 1 帧达到或低于 −100。最后连续 178 条指令（6.428 秒）全部请求 ≤0.1 mm 开度，对应的 1,285 帧设备反馈仍为 9.8 mm。弱负值确实存在，但正常闭合时出现的明显脉冲和随后的位移均未复现。表中 −200 及 0.3 mm 仅作标记时刻用，并非厂家阈值。

11 月 25 日按执行日志中最后一次连续进入闭合误差 ≥4 mm 的首条提交对齐，保证观察点在提交之后。14:22 当轮在 +182.5 ms 记录到 −593，+202.3 ms 首次观测到至少 0.3 mm 的闭合，最后从 57.5 mm 收到 49.4 mm。14:24 当轮从 64.8 mm 到 64.7 mm，记录中的最负值为 −87。这两轮使用本机提交和按时间戳去重的反馈，最大相邻反馈间隔分别为 25.14 和 30.00 ms，不能排除采样点之间有更短的瞬态。[11 月 25 日四轮开度对照](../piper_gripper_response_20261125/README.md)

## 需要厂商提供的内部读数

请松灵确认 S-V1.9-1 的夹爪控制器能否同时间戳导出 **最后采纳的目标开度及执行状态、转矩电流给定 Iq 和实测 Iq**，以及 0x2A8 力度字段的单位与来源。数据可分别核对目标是否写入、位置误差有没有形成电流需求、驱动电流是否跟随给定。公开 CAN 证据支持“停滞时反馈力度没有随闭合需求明显上升”，尚不足以判定固件丢指令、电机完全没有输出或夹指机械卡滞。

复算程序接受私有的三段原始记录窗口（按首条命令的内核实时时戳命名，32 字节定长记录，含监听器配置）、安装源码导出、官方源码目录和两份执行日志。窗口分别保留首条目标前 400 ms 至末条目标后 350 ms。公开统计包含输入窗口哈希与采样计数。[复算程序](audit.py) · [统计与证据索引](outcomes.json)
