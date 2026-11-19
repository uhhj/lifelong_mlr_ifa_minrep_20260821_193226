# S-V1.9-1 双臂直连的软件急停恢复核对

截至北京时间 2026-11-19 17:34，公开资料与源码核对完成。松灵公开资料能确认单台机械臂的角色切换、急停复位和回零后的拖动恢复。针对两台 PiPER 共用一个 USB-CAN、均已上电且持续直连的接法，查阅的资料未给出主臂暂停发送目标的操作、停发确认，以及从臂解除急停后交回跟随的顺序。此次核对全部在离线环境完成，未接入现场窗口或总线。

## 厂家说明和协议字段

| 核对项 | 公开资料给出的行为 | 对本次接线的含义 |
| --- | --- | --- |
| 固件与角色切换 | 旧版 SDK 项目的文档为 1.8-8 及以后固件指定 V3 接口。新版 pyAgxArm 将 1.8-9 及以后固件映射到 V189 驱动。维护者确认 1.8-9 已修复单臂从 Leader 切回 Follower 的问题。 | S-V1.9-1 在上述范围内。该答复讨论单台臂的工作角色，未给出双臂在线时的停发步骤。[V3 双臂说明](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/assets/double_piper.MD) · [固件映射](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/docs/piper/firmware_reference.md) · [维护者答复](https://github.com/agilexrobotics/pyAgxArm/discussions/53) |
| 桌面上位机解除急停 | 产品手册第 3.6 节描述再次操作急停按钮后重新使能。第 3.10 节描述主从角色和地址偏移配置。双臂 SDK 指南说明主臂发送关节、夹爪和模式指令，并在上位机独立控制从臂的示例末尾要求先断开主臂。 | 手册没有把再次按按钮列为双臂在这条总线上交接控制权的流程。[手册第 3.6 节](https://cdn.shopify.com/s/files/1/0551/0630/6141/files/PIPER_Product_User_Manual_EN_1.pdf?v=1794214904#page=43) · [双臂指南](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/assets/double_piper.MD#L99) |
| 角色配置与应答 | `0x470` 编码角色及反馈、控制和联动目标三个地址偏移。角色字节的零值定义为无效，其余两值分别选择示教输入臂或运动输出臂。`0x476` 通用应答解出指令索引和零点设置标志。 | 角色设置帧没有选定某台控制器的字段。修改后续报文的地址偏移不能证明主臂已经停止发送。通用应答没有停发状态或本次应答的主臂身份。[配置字段](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/transmit/arm_master_slave_config.py#L24) · [V3 编码](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/protocol/protocol_v3/piper_protocol_v3.py#L410) · [通用应答](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_set_instruction_response.py#L14) |
| 暂停与恢复 | `0x150` 的暂停字段分别管理已规划轨迹和拖动示教回放。`0x191` 的主臂恢复操作用于回零后回到零力拖动。V3 常量 `0x422` 的用途标注为 CAN 升级总线静默。 | 这些定义均未提供直连运行中单独暂停主臂目标并确认停发的操作。[轨迹字段](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/transmit/arm_motion_ctrl_1.py#L20) · [主臂恢复接口](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/pyAgxArm/protocols/can_protocol/drivers/piper/default/driver.py#L1329) · [V3 常量表](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v3/can_id.py#L108) |

## 离线与现场证据

[离线帧核对](check_v3_frames.py)在禁止创建 Python 套接字的进程内运行官方 V3 编解码器。81 种合法角色与地址偏移组合的配置帧 ID 全部为 `0x470`。测试还核对了停止、复位及两种轨迹暂停的不同字段，并解码了两个合成的 `0x476` 应答。在已安装上述提交的 piper_sdk 及 python-can 的环境可运行 `python check_v3_frames.py`。原窗口的 9 项离线 Qt 检查也已复跑通过。测试范围为报文格式和窗口逻辑。

此前现场的两组版本回包均为 `H-V1.2-1 / S-V1.9-1`。现场操作员按下 STOP 后，操作臂连续反馈急停，随后三秒内仍记录到三种主臂关节目标报文各 166 帧。该记录说明从臂报告急停与主臂继续发送目标可以同时发生。[现场观察与时间窗](README.md)

## 尚需厂商明确的问题

对于两台均为 S-V1.9-1 的 PiPER，保持供电、共用同一 CAN 总线和原角色配置，从臂已经反馈软件急停。是否有受支持的现场按钮或协议操作，可以仅暂停主臂关节、夹爪和模式指令？如有，哪些反馈可确认主臂已停发，从臂可在什么姿态条件下解除急停、重新使能，再恢复跟随？若固件另有无需停发即可恢复的原生流程，也需要相应的前置条件与反馈判据。手册第 5 节将完整 CAN 协议列为向技术支持索取的文档，官方联系方式为 `support@agilex.ai`。[手册第 5 节](https://cdn.shopify.com/s/files/1/0551/0630/6141/files/PIPER_Product_User_Manual_EN_1.pdf?v=1794214904#page=60) · [官方技术支持](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/README.md)
