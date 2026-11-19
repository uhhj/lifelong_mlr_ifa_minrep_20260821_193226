# PiPER S-V1.9-1 直连持位诊断与技术支持询问

## 可直接发给松灵技术支持

您好，我们的两台 PiPER 按官方方式主从直连，共用一条 CAN 总线，现场版本回包为 H-V1.2-1 / S-V1.9-1。两次都是主臂完全静止时从臂突然抽动。以下均为北京时间及 CAN 接收时刻。2026-11-18 21:14:04.715641，J2、J5 在同一轮反馈中先变，J2 转速从 0 到 −1.652 rad/s，电流从 +0.521 到 +1.177 A，下一条转速已反向至 +0.242 rad/s。2026-11-19 21:55:21.902717，J6 角度从 −6.772° 到 −6.529°，随后连续两条驱动反馈相隔约 10 ms，转速从 +1.434 到 −1.390 rad/s，电流从 +3.630 到 −0.111 A。两次起点附近均未收到新的主臂关节目标帧或 CAN 错误帧，六轴驱动状态均为 0x40。请问该固件在直连持位时，能否用厂家只读工具同步导出各轴实际采用的目标角、给定电流（Iq_ref 或等效量）、电流反馈，以及原始编码器读数和故障计数？0x2A5–0x2A7 的关节角与 0x251–0x256 的电机位置是否来自同一传感器？设备现已断电，请告知是否保留可导出的故障记录、相应工具或协议、采样周期，以及如何用这些数据区分编码器异常、持位伺服异常和瞬时掉压。我们留有两次事件的原始 CAN 日志，可按需提供。

## 官方资料核对

松灵的双臂说明为 1.8-8 及以后固件指定 `C_PiperInterface_V3`；新版 pyAgxArm 的固件表把 S-V1.8-9 及以后归入 `PiperFW.V189`。S-V1.9-1 落入这两个范围。下列结论限于公开接口，厂家专用诊断能力须由技术支持确认。[官方双臂说明](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/assets/double_piper.MD) · [固件适配表](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/docs/piper/firmware_reference.md)

| 想读取的量 | 公开接口能核实的内容 |
|---|---|
| 从臂内部持有的目标角 | `GetArmJointCtrl()` 读取主臂发出的 0x155–0x157；`GetArmJointMsgs()` 读取从臂关节角反馈。`get_cpv_pos()` 的文档将其定义为 CPV 通道的关节位置反馈；`get_ik_joint_angles()` 对应末端位姿指令触发的逆解。未找到直连持位时回读内部实际目标角的公开定义。[双臂数据读取](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/assets/double_piper.MD) · [CPV 读取](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/pyAgxArm/protocols/can_protocol/drivers/piper/default/driver.py#L2758-L2789) · [逆解反馈](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/interface/piper_interface_v3.py#L378-L402) |
| 驱动器给定电流 | 0x251–0x256 携带电机转速、电流和位置反馈。`current` 对应反馈电流。公开消息没有单列伺服内部的给定电流，高速帧里的电流值能证明电流反馈变化，不能确定是哪一环先触发。[高速反馈定义](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_high_spd.py) |
| 编码器专项诊断 | 可读取电机位置和驱动汇总状态。0x261–0x266 的状态字节 bit 4 现为碰撞保护，bit 5 为驱动错误，bit 6 为使能。未找到编码器原始采样、信号质量或丢码计数的公开定义。现场的 0x40 表示该帧只有使能位置位，不能据此排除反馈链异常。[状态位定义与历史语义](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_low_spd.py#L33-L42) |

两次时间及数值取自已归档的被动监听，日志所给的是帧到达时间。本次核对只使用公开资料、SDK 源码和既有记录。[首发帧和基线](README.md) · [反向、电流和供电对照](differential.md) · [现场版本回包及共线身份限制](../piper_direct_resume_offline_20261119/README.md)
