# 仅C等生产optimizer预算实现

用户在2026-10-30选择保留完整S预算并保持各自方法，随后明确只重跑C。PROTOCOL.md取代历史PROTOCOL_DRAFT；新run仅六C，已有六S保持只读，不调用任何S train/collect/evaluate。独立root与输出防重部署。

prepare_local以已接受五C扩展源码为基础，GOAL41取GOAL42同参数化实现并绑定GOAL41自己的官方source/data/seed，绝不复用42初始化。原updates/losses/evaluate等逐字不变，source hash/diff见UPLOAD_REVIEW及review。Task6 100轮，后续200轮；显式更新cfg.train.n_epochs和a.total_steps后构造原linear scheduler，保持warmup500。新增实际schedule检查防止旧horizon截断；每epoch记录两类物理step计数。

AdamW.step只包装原调用后增加Python整数计数，不改变输入/返回/随机流；计数不会被state restore回滚，因此计入隔离重放。原C各stage准入4/5保留，后三阶段另24调用执行10步原版/legacy完整重放及保存加载后的2步重放，总4/29，与已有S自身4/29相等，所有诊断结束恢复初始完整state。初始模型/RNG/hardware与已接受S完全比较；Task6最终所有tensor与实际S stage6 CP比较；未来实际PASS不得由本地静态替代。

复制优化candidate与已验证OBJECT工程版本相同，只去除CUDA→CPU后冗余clone、保留CPU clone。每阶段真实capture与旧cpu实现逐项比较全部state。数据/模型不改变，性能改善不做额外宣称。

LOCAL_BUILD_REVIEW验证六C逐阶段预算等于实际接受S EPOCHS物理预算，S_REFERENCE和S_ACCEPTANCE绑定已接受EVIDENCE/LOCAL_FINAL_AUDIT/INITIAL/TRAIN/STAGE/OUTCOMES哈希。首次本地parent原始字节比较因Windows CRLF与远端LF差异失败；改为实际传输UTF8 LF规范化后六文件hash精确匹配，此本地错误没有远端optimizer/environment调用。

final_readonly从已验收C真实CPU包装外置适配六case、700轮/200最后epoch、新恢复计数和已有S Task6 tensor检查。audit_progress按100/200/200/200分段、schedule/copy/隔离与formal事件noise校验；final local实际独立复算全部24CP/18cache/scalars/6recovery/1200回合后，compare_primary_s核已有S每阶段预算、共同初态和1200回合字节noise配对。FINAL_WRAPPER_REVIEW仅STATIC_ONLY，代码存在不等于验收通过。

每个阶段C消耗的全部生产更新均保留，而S从双候选只保留一个；scheduler按各自保留轨迹长度衰减。等物理optimizer不等每次调用LR、FLOPs、墙钟、额外数据或交互。C0采集；已有S共240采集。旧S工程与中断未记账成本保持独立披露。


## 2026-10-30 16:20 用户调度修订（覆盖此前全训练后统一评估顺序）

GOAL41/42/43训练→GOAL41/42/43正式评估→OBJECT41/42/43训练→OBJECT41/42/43正式评估。S只读，所有科学预算与配置不变。不打断现有GOAL42或重跑完成部分。工程目录engineering_suite_order_20261030_v1实现自然训练边界交接，保持原科学冻结源；工程完成标记为该目录controller.exit，原root/controller.exit不伪造。详细安全证据、使用方式与当前ARMED/完成状态以MONITOR_HANDOFF及实际快照为准。


## 2026-10-30 用户要求先独立验收GOAL

GOAL三个C已自然完成训练和600正式回合。用户明确要求先验收GOAL，故新增goal_acceptance_20261030_v1外置只读包装：以GOAL_EVALUATION_COMPLETE和各GOAL自身train/evaluate.exit0、无GOAL进程为完成门槛，允许原OBJECT继续。只保留原GOAL三case CPU检查，原检查内容及容差不变；独立低优先级单线程CPU，审计前后GOAL证据SHA一致。新状态PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE明确仅GOAL接受，不伪造原root/controller.exit、不修改全六C最终包装、不把OBJECT未完成算作接受。没有重训、重评、环境回合或优化器调用。全六C完成后仍执行原工程最终验收。
