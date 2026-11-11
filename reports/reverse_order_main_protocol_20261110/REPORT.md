# 主表协议逆序实验

> 协议审查暂停新增派发：controller119466已单独SIGSTOP；现有训练继续，本地观察器已停止，旧下一检查时间失效。不得自动CONT、重启或追加计算。详见PROTOCOL_REVIEW_HOLD.json及前序实验PROTOCOL_SCOPE_REVIEW.md。

PARTIAL，实际观察2026-11-11T07:45:06.106314+08:00。只读审计PASS_PROGRESS_NOT_FINAL，不代表最终接受或局部性能结论。

原基底0–5和官方检查点不变，增量物理任务9→8→7→6。MLR+IFA和完整DEBR分别继承主表训练与原生评估实现。

|case|完整epoch|接受更新|生产物理更新|训练采集|正式|
|---|---:|---:|---:|---:|---:|
|OBJECT_MLR_41|400|102100|102100|0|0|
|OBJECT_DEBR_41|153|24966|36732|10|0|
|OBJECT_MLR_42|90|11880|11880|0|0|
|OBJECT_DEBR_42|0|0|0|0|0|
|OBJECT_MLR_43|0|0|0|0|0|
|OBJECT_DEBR_43|0|0|0|0|0|
|GOAL_MLR_41|0|0|0|0|0|
|GOAL_DEBR_41|0|0|0|0|0|
|GOAL_MLR_42|0|0|0|0|0|
|GOAL_DEBR_42|0|0|0|0|0|
|GOAL_MLR_43|0|0|0|0|0|
|GOAL_DEBR_43|0|0|0|0|0|

下一检查2026-11-11T08:36:06.106314+08:00，范围CURRENT_WHOLE_RUN_TRAINING_AND_COLLECTION；估计含明确规划余量，不是全队列保证结束时间。最终48CP/12恢复/24缓存与240训练2400正式字节及预算尚待实际验收。
