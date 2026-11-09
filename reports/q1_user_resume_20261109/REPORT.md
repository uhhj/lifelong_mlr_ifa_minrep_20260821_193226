# 扩容重启后的核验与续训

观察时间：2026-11-09T18:55:20.027568+08:00。状态 PARTIAL_RESUMED_AFTER_EXPANSION_REBOOT，不是最终实验接受。

数据盘已扩容，可用 44.83 GiB；系统盘可用 1.32 GiB。/data 为原 /dev/vdb ext4 可写挂载。使用 available 而非包含文件系统保留块的 free；当前 q1 剩余输出上界约 4.84 GiB，加保留量及后续 ALIGN 预留均能容纳。未来部署仍须新鲜预检。

扩容期间主机重启，原六暂停进程全部消失；未发送 SIGCONT 或重新执行旧 controller。148 原科学冻结项、原 epoch/gate/cache 绑定均保持。GPU RTX 4080 SUPER 驱动可用。

两个原 checkpoint 完整可读，分别保存 347 和 340 轮。实际 GPU 完整 model/optimizer/scheduler/grad/mode/requires/RNG/step 恢复及序列化回放精确；Task9 scale、probes 和各自缓存绑定精确。原未保存部分轮的 10/17 个门控块逐字段精确重放，验证 200+340=540 optimizer、0 environment、0生产轮，原训练前缀未改。

验证加载原 wrapper 时写了两个 RENDEZVOUS 通信元数据，首次部署严格暂停 SHA 检查因此在创建备份/启动前停止。原值从既有快照按暂停 SHA 精确找回，原/验证值均外置保存；v2只精确绑定这两个已解释的元数据变化，其余检查不放宽。没有重复 GPU 验证。一次本地生成器语法错误及一次UTF8读取错误仅影响本地准备，没有训练或环境成本；均在实际部署前/包装生成时修正。

唯一实际部署 1794221617.0645158，新工程合同 9c5b7e0b23e1beabb4bf28216f6a75f372d6d2f68b3f6515c27debdd8eda34bb、7项，新总 controller2416、OBJECT controller2417、GPU2430/2428。原 checkpoint 以服务器同文件系统硬链接备份保留，不下载模型；后续 atomic save 会保留原恢复点。源主循环科学候选/探针/选择/恢复代码未改，完整轮不重跑，仅重建未保存轮并核对既有门控行，不重复追加。生产恢复再次逐字段精确，已新增一完整轮，各为348/341。独立进度审计 PASS，之前所有完整 epoch/gate 前缀精确保留。

额外成本必须分开：隔离验证540次；原中断轮已记录候选调用540次，最后未记录部分 UNKNOWN，不能称总物理成本精确。恢复运行中的前缀重建属于最终规范序列，原被替代调用作为额外成本保留。暂停/重启和诊断墙钟单列；不将旧慢速或暂停间隔混作正常epoch速度。

有限队列仍为 OBJECT三个seed最多两训练→600正式→GOAL同序，所有环境互斥，旧S只读。原全局/OBJECT controller.exit 保持不存在；成功来源只用新工程真实退出。OBJECT最终包装已外置适配（STATIC_ONLY，未执行），GOAL原包装保持。最终两套件实际CPU接受/比较/发布后，再以版本化真实完成源适配已准备 ALIGN 的启动门槛，不伪造旧exit、不提前启动ALIGN。
