# PiPER 官方直连状态

PARTIAL，2026-11-18 21:07:57 北京时间。上位机在一条 CAN 总线上持续收到执行臂反馈，现场完成直连接线后，尚未在此观察窗口确认主臂目标到达或动态跟随。

- 上位机仅识别一个 USB-CAN。20:57 接口处于 DOWN/STOPPED，9 秒内收发计数均为零。为保留现场监听，将监测口 `can0` 配为 1 Mbit/s 硬件只听，核实 LISTEN-ONLY 后启用接口，启用后再次核实。查询期间没有启动机械臂或转发程序。
- 21:00:51.406 至 21:01:00.407 的全 ID 采样收到 21,790 帧，接口发送数、收发错误、接口丢包和 CAN 错误帧均为零。六轴角度在此窗口内逐值不变，J1 至 J6 分别为 0.605°、0.106°、0.124°、−3.889°、22.933°、14.113°。
- 六轴驱动状态在此窗口均为 0x40，即已使能且未报告故障。最后一组低速反馈的电压为 23–24 V，驱动温度 34–39 °C，电机温度 28–35 °C。
- 复核常驻监听的 21:00:20.329 至 21:07:57.388 记录。91,412 条整臂状态均为 `0100010001000000`，表示 CAN 指令控制、整臂状态正常、MOVE J、运动状态位为“未到达指定点位”、故障码为零。窗口内没有原生模式、关节或夹爪目标帧（0x151、0x155–0x157、0x159），没有错误帧。截取的是总线观察窗口，缺少现场动作标记。
- 核对相关进程时，原被动监听器正在记录 can0，未发现旧转发进程。保留硬件只听供后续现场操作对时。这个端口只能观察，原转发进程的 500 ms STOP 不在直连指令路径内。

整臂状态和驱动位按厂家协议解码。 [AgileX V3 状态字段](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/piper_msgs/msg_v3/feedback/arm_feedback_status.py) · [驱动状态位](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_low_spd.py) · [官方双臂连接及报文](https://github.com/agilexrobotics/piper_sdk/blob/master/assets/double_piper.MD)
