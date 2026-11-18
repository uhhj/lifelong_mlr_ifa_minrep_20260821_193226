# PiPER 试采中断与官方录制工具核查

2026-11-18 16:26（北京时间），核查一轮已经结束的现场试采。服务器检查仅读取日志、包索引和源代码。

## 试采中断

- 16:23:09 的中间状态（PARTIAL）为 `bound_waiting_input`，已收关节组和夹爪指令均为 0，录制入口处于禁用状态。16:23:34 状态变为 `following`，已收关节组和夹爪指令各 283，两路相机均已打开。
- 本轮以 `native joint timestamp moved backwards` 中止，从臂反馈确认 STOP。回查整轮主臂关节输入，定位到相邻 `0x156`、`0x157` 帧间唯一一次投影时间倒退。该帧对的原始内核时间差为 +5,376 ns，接收侧单调时钟差为 +185,864 ns，逐帧投影得到的单调时间差为 −4,633 ns。控制组帧使用了投影时间的单调性保护，因此触发 STOP。
- 一条试采记录保存了 43 对图像和 496 条动作流记录，标记为 `aborted`，不可用于训练。审计时正式示范目录中有 0 条记录。两个 CAN 口快照均为 `ERROR-ACTIVE`，`bus-off` 计数为 0。

## 官方采集方案

- AgileX 的 `data_tools` 有 ROS 2 分支，可通过 YAML 指定相机和关节话题。采集节点订阅这些话题并发布自身采集状态。测试机的 ROS 2 Humble 和已检查的两个桌面工作空间中均未找到该包。现有 CAN 遥操输出需要先提供只读 ROS 2 数据源，供录制工具接收顶视与腕视 RGB、实际发送目标及从臂反馈。还需验证静止起录和持续持有同一目标时的记录完整性。官方启动参数支持设置采集频率、等待时长和裁剪时长。[配置和用法](https://github.com/agilexrobotics/data_tools/tree/ros2) · [采集节点](https://github.com/agilexrobotics/data_tools/blob/ros2/src/dataCapture.cpp) · [启动参数](https://github.com/agilexrobotics/data_tools/blob/ros2/launch/run_data_capture.launch.py)
- Hugging Face 提供 LeLab 网页录制界面和 `lerobot-record` 命令，PiPER 由第三方接入。测试机已安装 LeRobot 0.5.2 和录制命令，现有 PiPER 模块为本机附加文件。实读本机录制循环发现会调用 `robot.send_action()`，该 PiPER 从臂后端会写 CAN。因此沿用现有遥操时，首选订阅数据的采集方案。[LeLab](https://huggingface.co/docs/lerobot/v0.6.0/en/lelab) · [PiPER 插件列表](https://huggingface.co/docs/lerobot/en/third_party_robots)

本轮完成故障定位与官方工具的静态兼容性检查。实机接入尚未验证，机械臂未远程启动，500 ms 的 STOP 保护仍在。
