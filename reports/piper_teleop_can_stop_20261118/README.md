# PiPER 现场短测 CAN 保护停机复核

复核 2026-11-18 14:18（北京时间）的遥操短测。运行退出原因为 `Abort('CAN error frame')`，操作臂 STOP 反馈已确认。程序发现 CAN 错误帧便进入既有保护路径。

- 从逐帧时间线复算，动作段 18 组主臂六轴目标均按原值转发。组内跨度最大 0.3249 ms，动作段首尾约 109.91 ms。76 条目标发送记录均有相符的 `can0` 发送回显，发送失败记录为 0。此轮未触发分包时窗或补齐超时。
- 事后两个接口均为 `UP` / `ERROR-ACTIVE`，`bus-off` 为 0。操作臂 `can0` 的累计 `error-pass` 为 258,037，主臂 `can1` 为 0。空闲状态被动监听 3 秒未读到新的错误帧，两路相关计数未变化。应优先检查 `can0` 链路，累计值无法单独定位本次错误帧。
- 现用 `transport_base.py` 在逐帧记录前对 `is_error_frame` 终止，旧运行记录缺少触发帧的接口、错误掩码和数据，故障现场的物理原因待确认。下一轮现场短测应同步被动记录两路错误帧和接口计数，区分 ACK、控制器状态变更及协议错误。保留现有 STOP 行为。 [SocketCAN 错误帧与接口统计](https://docs.kernel.org/networking/can.html#network-problem-notifications)
