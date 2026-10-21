# LIBERO50 taskmap01工程恢复

原共享预训练在2026-10-22 05:07的GPU准入中失败，尚在pretrain_dataset加载，未到CompletePolicy构建、优化器创建或更新。旧输出仅CONTRACT、resolved_pretrain与RUN_STATUS三项，GPU与目标进程均退出。原controller.exit1、entry.exit1、gpu_preflight.exit1和capacity watcher拒绝非容量错误的exit1均保留。

只读原源码/输入408项hash全通过。旧LIBERO90构造器保留筛选后的任务，冻结50任务中有6项SCENE9被遗漏，其中预训练首个漏项是put_the_frying_pan_on_the_cabinet_shelf。完整task_maps可以精确解析50/50，原44项的Task全部字段一致，BDDL/初态/dataset相对路径逐名一致，未修改分区或任务文本。

新taskmap01 bundle仅修改l50_exp/data.py的native_task，改为完整task_maps精确键查询并断言三类路径。原196文件中的195项字节不变。原bundle及冻结CONTRACT不修改。新bundle SHA为625febe8e1541fecc041b367aa7cdc77afec41103c7aa9c5f5ed1a06b57638b7；原bundle370dd464...保持。新映射源50身份检查PASS，零模型forward/优化器/环境，CUDA未初始化，不能作为GPU通过。

恢复调度位于Q/libero50_taskmap01，只含L50共享及41→42→43，原CAET完成前缀receipt逐项绑定。共享新输出/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021_taskmap01；三个下游原路径追加_taskmap01，41仍系统盘，42/43仍data。原失败输出原样只读留存。100epoch、batch10、FP32、1,941,000共享更新+三seed各859,400更新及4500正式预算全部不变。

控制器持原queue.lock，验证新旧合同，空GPU，系统仅普通空闲≥4.5GB，data root总空闲≥4.5GB+2GiB，不改文件系统、不删资产。每入口自己执行真实GPU准入；失败停机留证，不盲重试。最终合并原CAET前缀与新L50后缀，旧exit1不可改写为0。
