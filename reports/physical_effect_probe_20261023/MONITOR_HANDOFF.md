# 当前交接
当前仅新PEP，不复活旧队列。先读USER_METHOD/PROTOCOL/REPORT/RUN_STATUS及ACTIVE_ROOT。SSH用remote.py或check_progress.py，凭据getpass仅内存。现阶段A训练100fits，不重启健康进程。
check_progress.py保存一次只读快照；audit_progress.py <snapshot>本地核身份/整演示拆分。完整A之后须独立复算全部30种任务/模型/域组合、100fit×50、输入SHA、gate；不能凭服务器标志直接接受。
A不通过则正确停止B，不自动换特征/放松门槛。A通过才实现并审计B独立版本；控制3臂、300episode、预测物理输入和条件扩展按PROTOCOL。A权重不能进入B。
每轮使用publish.py经publish_mlr_branch精确审查/非force推MLR并核远端。只发布代码报告标量误差/哈希，禁止private目录及二进制；inspect_inputs原始inventory含旧动作字段仅本地保留已排除发布。旧A/N不重复发布。失败先核进程停/hash一致留证版本化修，不修改被冻结源，不重试科学负结果。
