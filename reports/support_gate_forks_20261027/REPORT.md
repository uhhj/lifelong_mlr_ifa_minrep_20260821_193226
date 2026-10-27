# 支持度与门控共同起点机制实验

状态：PARTIAL_TRAINING；观察 2026-10-28T00:09:42.653404+08:00。

已实际接受完整epoch账目 121；主更新 22385；候选实际训练优化器调用 22385；隔离准入 58；正式环境 0/1120。尚无科学收益或最终验收结论。

{
  "GOAL_41": {
    "C": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 8.987632513046265
    },
    "S": {
      "epochs": 21,
      "accepted_updates": 3885,
      "physical_optimizer_calls": 3885,
      "last_epoch_seconds": 14.611109256744385
    },
    "Q": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "G": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    }
  },
  "GOAL_42": {
    "C": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "S": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "Q": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "G": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    }
  },
  "GOAL_43": {
    "C": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "S": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "Q": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "G": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    }
  },
  "OBJECT_41": {
    "C": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "S": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "Q": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "G": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "U": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    }
  }
}

{
  "case": "GOAL_41",
  "arm": "S",
  "observed_epochs": 21,
  "recent_epoch_seconds": 14.174384188652038,
  "remaining_training_minutes": 18.662939181725182,
  "preparation_saving_extra": true,
  "not_total_queue_prediction": true
}

首组GPU真实准入已证明lambda=0与原C完整更新相等、q=1与原S完整更新相等、10步C重放、完整恢复及探针/梯度测量不改变训练状态；其余组必须各自通过。额外候选和探针计算单列，不称等算力。

机制实验保留C/S/Q/G四组与OBJECT41的U开发对照。之后仅最新完整S运行OBJECT41/42/43完整序列，旧方法不新增运行；不得将机制阶段结束误作用户全部任务完成。参见PROTOCOL和IMPLEMENTATION_NOTES。

本地一次辅助编辑命令因PowerShell引号产生Python解析错误，未执行目标修改；部署仍使用完整已审科学PROTOCOL，未改科学源码或预算。最终外置wrappers仅编译，须实际运行验收。
