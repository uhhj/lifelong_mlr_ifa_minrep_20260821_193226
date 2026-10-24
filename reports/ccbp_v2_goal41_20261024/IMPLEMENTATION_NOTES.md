# 实现说明

从已验收v1 runtime/controller/evaluate派生，新目录新合同。科学修改只在N/V辅助loss；B/C逐损失/梯度及GPU完整更新与旧实现比较。上传原型原字节保存在prototype_reviewed；prototype_loss.py仅将运行时类型别名tuple[...]改为typing.Tuple[...]以兼容服务器Python3.8，公式不变。首次无写入CPU探针在导入类型别名时失败，0模型前向/更新/环境，ATTEMPT保留。新的快速固定配置实现与兼容原型进行合成值/梯度精确对照；原型uniform模式不形成新实验组。

梯度诊断是固定epoch的额外反向计算，不增加optimizer调用；必须通过开关诊断不影响下一步状态的自身准入。teacher cache旧430窗口原地只读复用，新增tau与w为训练标量，完整绑定CP/cache/label/manifest。

四组全部重训共同Task9分叉，不复用B/C已训练权重。新的最终CP与恢复文件共享同一完整状态的磁盘inode，不是删除权重或缩减恢复态；序列化使用storage别名避免重复写model，原子替换发生在最终硬链接创建之前。

初次本地生成CPU探针的PowerShell内联字符串转义失败，未生成脚本；等待口令的runner已在认证前取消，无服务器命令。改用独立build_cpu_probe.py生成，工程过程留证，不算科学训练重试。

当前静态实现尚待自己的完整GPU准入与正式运行。所有旧测试PASS均不得套用本轮。
