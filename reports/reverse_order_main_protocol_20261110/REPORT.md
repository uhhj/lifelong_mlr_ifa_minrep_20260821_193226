# 主表协议逆序实验：派发暂停下的只读进度

实际观察2026-11-11T09:10:43.317711+08:00，PARTIAL。独立轻量审计PASS_PROGRESS_NOT_FINAL。controller119466实际状态T，未来派发保持暂停；在训子进程未停止，未自动恢复派发。

基底任务与官方预训练不变，增量原物理任务9→8→7→6；两方法各自继承主表原生训练和评估实现，未在本次检查改协议。

|case|实际运行状态|完整epoch|接受更新|训练采集|正式|
|---|---|---:|---:|---:|---:|
|OBJECT_MLR_41|TRAINING_COMPLETE|400|102100|0|0|
|OBJECT_DEBR_41|ACTIVE_TRAINING|252|50584|20|0|
|OBJECT_MLR_42|TRAINING_COMPLETE_DISPATCHER_EXIT_REAP_PENDING|400|102100|0|0|
|OBJECT_DEBR_42|NOT_DISPATCHED|0|0|0|0|
|OBJECT_MLR_43|NOT_DISPATCHED|0|0|0|0|
|OBJECT_DEBR_43|NOT_DISPATCHED|0|0|0|0|
|GOAL_MLR_41|NOT_DISPATCHED|0|0|0|0|
|GOAL_DEBR_41|NOT_DISPATCHED|0|0|0|0|
|GOAL_MLR_42|NOT_DISPATCHED|0|0|0|0|
|GOAL_DEBR_42|NOT_DISPATCHED|0|0|0|0|
|GOAL_MLR_43|NOT_DISPATCHED|0|0|0|0|
|GOAL_DEBR_43|NOT_DISPATCHED|0|0|0|0|

训练完成记录与optimizer计数完整不等于最终科学验收。MLR42的父派发器退出回收尚待进行，不把未记录的launcher退出码伪称0。正式评估尚未启动。

在训任务ETA（仅当前子进程，未测阶段按最近实际每batch速度外推；含明确规划余量，不含正式评估／最终验收）：

{
  "OBJECT_DEBR_41": {
    "pure_epoch_minutes": 158.10933963324925,
    "planning_allowance_minutes": 20,
    "minutes": 178.10933963324925,
    "estimated_CST": "2026-11-11T12:08:49.878089+08:00",
    "last_ten_seconds_per_epoch": 53.771170711517335,
    "remaining_stages": [
      {
        "stage": 8,
        "original_task": 7,
        "complete_epochs": 52,
        "remaining_epochs": 48,
        "seconds_per_epoch": 53.771170711517335,
        "unmeasured_stage_extrapolation": false
      },
      {
        "stage": 9,
        "original_task": 6,
        "complete_epochs": 0,
        "remaining_epochs": 100,
        "seconds_per_epoch": 69.05544183842123,
        "unmeasured_stage_extrapolation": true
      }
    ],
    "not_guaranteed": true,
    "excludes_formal_and_final_acceptance": true
  }
}

GPU与容量见RUN_STATUS。无新失败记录，冻结源及旧证据前缀核验通过。本次0新增优化器调用／环境／恢复派发。没有设定下一自动查询；不重启本地观察器。
