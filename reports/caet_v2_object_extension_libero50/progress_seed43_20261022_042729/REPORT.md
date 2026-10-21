# CAET-v2 OBJECT43评估进度

状态PARTIAL，观察时间2026-10-22T04:28:08.874663+08:00。

最新2026-10-22 04:28 CST：progress_seed43_20261022_042729，43正式80/200条、8完成vector+1进行中，evaluate76560及10环境正常、0技术异常。400epoch105500主步、Task8/9迁移7612/9036与保护2578/4918不变；505冻结hash与逐epoch/逐步及逐episode初态/attempt成功向量独立PASS。8批均255.10秒，剩评估约51分钟加最终验收，预计05:25–05:50 CST参考。无重启，无部分指标；cell summary/4CP严格加载/有效退出仍待最终包。队列仍43、容量watcher仍ARMED，data普通7.906GB/root12.776GB、system普通5.902GB，LIBERO50未GPU准入。外置check_queue.py额外读取PRETRAIN_INITIALIZATION.json，便于随后新预训练真实窗口/批次/可训练模块身份核验；只读字段扩充，不改服务器科学源。已只读确认冻结pretrain.py每完整epoch记录单调时钟耗时，首epoch前不能伪造吞吐；初始化不是GPU或完成验收。三seedaggregate_object.py仍仅静态语法PASS，须43最终验收后才运行。
