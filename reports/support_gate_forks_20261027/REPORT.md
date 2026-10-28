连接复查 2026-10-28T09:57:42.421798+08:00：SSH仍在认证前失败，远程命令0，当前实验状态未知。已尝试查找可用服务器管理页面，但电脑操作工具初始化因本机kernel assets路径缺失失败，未访问或修改服务商页面。worker4572保持，原训练未操作；错误SHA见connectivity_readonly_20261028_0957/LOCAL_AUDIT.json。继续60分钟传输复查。

08:55:32 CST复查：SSH仍在认证前失败，remote_commands=0，worker4572未重启。最新训练观察仍为07:19:17，当前状态未知；原错误保留在check_readonly_20261028_085527/ATTEMPT.json，新绑定见connectivity_readonly_20261028_0855/LOCAL_AUDIT.json。将传输复查退避为60分钟，旧训练ETA已标失效。

本轮连接状态：PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE（2026-10-28T08:24:32.285406+08:00）。两次SSH在认证前失败，远程命令0；独立端口检查TCP连接成功但读取0字节即断开，未返回SSH banner。不能据此判断训练失败或停止。以下07:19:17为最后成功观察，非当前进度；未重启worker/训练、未改参。原失败证据和SHA见connectivity_readonly_20261028_0820/LOCAL_AUDIT.json。30分钟后复查传输，恢复后先检查冻结hash、进程和原账本，不盲重启。

# 支持度与门控共同起点机制实验

状态：PARTIAL_TRAINING；观察 2026-10-28T07:19:17.083839+08:00。

已实际接受完整epoch账目 1304；主更新 259648；候选实际训练优化器调用 315148；隔离准入 406；正式环境 0/1120。尚无科学收益或最终验收结论。

{
  "GOAL_41": {
    "C": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 8.987632513046265
    },
    "S": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 15.96706748008728
    },
    "Q": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 14.320943355560303
    },
    "G": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 37000,
      "last_epoch_seconds": 37.433425188064575
    },
    "gate_blocks_observed": 1900,
    "gate_Q_accepts": 1085
  },
  "GOAL_42": {
    "C": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 9.262686491012573
    },
    "S": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 15.774545192718506
    },
    "Q": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 14.738142251968384
    },
    "G": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 37000,
      "last_epoch_seconds": 43.64219784736633
    },
    "gate_blocks_observed": 1900,
    "gate_Q_accepts": 764
  },
  "GOAL_43": {
    "C": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 8.96446180343628
    },
    "S": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 15.5825777053833
    },
    "Q": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 14.856539487838745
    },
    "G": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 37000,
      "last_epoch_seconds": 44.89176559448242
    },
    "gate_blocks_observed": 1900,
    "gate_Q_accepts": 1111
  },
  "OBJECT_41": {
    "C": {
      "epochs": 100,
      "accepted_updates": 36200,
      "physical_optimizer_calls": 36200,
      "last_epoch_seconds": 17.098241329193115
    },
    "S": {
      "epochs": 4,
      "accepted_updates": 1448,
      "physical_optimizer_calls": 1448,
      "last_epoch_seconds": 31.040600299835205
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
  "case": "OBJECT_41",
  "arm": "S",
  "observed_epochs": 4,
  "recent_epoch_seconds": 31.42976540327072,
  "remaining_training_minutes": 50.28762464523315,
  "preparation_saving_extra": true,
  "not_total_queue_prediction": true
}

首组GPU真实准入已证明lambda=0与原C完整更新相等、q=1与原S完整更新相等、10步C重放、完整恢复及探针/梯度测量不改变训练状态；其余组必须各自通过。额外候选和探针计算单列，不称等算力。

机制实验保留C/S/Q/G四组与OBJECT41的U开发对照。之后仅最新完整S运行OBJECT41/42/43完整序列，旧方法不新增运行；不得将机制阶段结束误作用户全部任务完成。参见PROTOCOL和IMPLEMENTATION_NOTES。

本地一次辅助编辑命令因PowerShell引号产生Python解析错误，未执行目标修改；部署仍使用完整已审科学PROTOCOL，未改科学源码或预算。最终外置wrappers仅编译，须实际运行验收。
