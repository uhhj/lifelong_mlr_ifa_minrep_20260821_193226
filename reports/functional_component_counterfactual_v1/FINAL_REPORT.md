# 功能子空间更新归因实验 v1

{
  "MEASUREMENT_VALIDITY": "PASS",
  "PROJECTED_UPDATE_ENGAGEMENT": {
    "median_rho_head": 0.004614147494750966,
    "median_rho_all": 0.0004112624967662666,
    "below_physical_resolution": 0
  },
  "LOCAL_EFFECT_REPLICATION": "SUPPORTED_ON_DEVELOPMENT_SET",
  "GEOMETRY_ADDED_VALUE": "NOT_ESTABLISHED",
  "BEHAVIORAL_EFFECT": {
    "status": "NOT_RUN"
  },
  "RESOURCE_COST": "MEASURED_SEPARATELY",
  "completed": 36,
  "formal_training_resumed": false
}

## 测量对象与范围

本轮测的是给定回放与AdamW状态的新任务条件增量，不是永久知识复用。36个epoch30–99起点来自三个开发seed和6→7、7→8两个阶段。完整MLR与移除新任务损失候选保留原分母、clipping、moments和衰减；本机原生candidate parity与回滚通过。主干预只改policy head，解码器保留native候选值。

## 能量与物理实现

介入比例见ENGAGEMENT.csv，top8/16/32完整trace覆盖见GEOMETRY_CHECKS.csv。E_full使用完整算子，尾部与交叉项保留；空间外不称安全。实际FP32 head逐候选写入，缓存来自各自BASE/R/M解码器；量化零效应保留，不做QP物理门。

## 跨轨迹效应和几何增量

{
  "D_T": {
    "old": {
      "spearman": 0.5680823680823681,
      "CI95": [
        0.1636431910903216,
        0.7867277094902351
      ],
      "seed_spearman": {
        "45": 0.7482517482517484,
        "46": 0.4895104895104896,
        "47": 0.46153846153846156
      }
    },
    "new": {
      "spearman": 0.72998712998713,
      "CI95": [
        0.39193194999355585,
        0.8441722596149626
      ],
      "seed_spearman": {
        "45": 0.6433566433566434,
        "46": 0.8531468531468532,
        "47": 0.6363636363636365
      }
    }
  },
  "paired_gain_intervals": {
    "D_T_joint_trajectory_block_CI": {
      "old": [
        0.1636431910903216,
        0.7867277094902351
      ],
      "new": [
        0.39193194999355585,
        0.8441722596149626
      ]
    },
    "GN": [
      -0.0009931196711908277,
      0.009389639242143222
    ],
    "GN_minus_C2_simultaneous97_5": [
      -0.001735367023170263,
      0.010495960353249904
    ],
    "GN_minus_C3_simultaneous97_5": [
      -0.0017211207437645513,
      0.010230812517784526
    ],
    "replicates": 5000,
    "scope": "Conditional on six fixed carriers; shared trajectory exponential bootstrap weights and length2 circular start blocks; three development seeds, no population generality"
  }
}

边际KEEP−REMOVE与各自相对BASE的绝对效应分开。X00/X10用于交互分解，详见INTERACTIONS.csv。D选择在T前冻结；GN固定HALF/REMOVE单列，三个随机重复取平均而非择优。六cell、三个seed及leave-one-seed-out均保留。未建立几何增量时不加rank/新几何。

## 尾部、行为与成本

逐任务/轨迹原始NLL、时间三分段及夹爪指令变化窗口见各start CSV与STATE_AND_TRAJECTORY_STRATA.csv；不称接触真值。旁观旧任务不建空间，单列其损伤。完整GMM KL估计保留负值和MC标准误，不能当利益标签。行为仅由BEHAVIORAL_REPORT.json中的实际480预算rollout判断，NLL不替代成功率；20次/任务低功效不证明等效。资源逐起点记录并披露T额外缓存访问，见RESOURCE_COST.json。

## 下一步

当前未建立GN超越等幅廉价对照的增量，不推进该几何到正式训练；不追加变体补救。

formal_training_resumed=false。

## 逐项数值回答

以下是冻结规则下的描述统计，不新增通过门槛。

### 2–3：空间介入、完整算子能量及截断外能量

```json
{
  "rho_head": {
    "n": 36,
    "mean": 0.004884984246712591,
    "median": 0.004614147494750966,
    "min": 0.0014204393941642621,
    "max": 0.011668956546638853
  },
  "rho_all": {
    "n": 36,
    "mean": 0.00044883988732813174,
    "median": 0.0004112624967662666,
    "min": 7.74720821613955e-05,
    "max": 0.0011894837881700946
  },
  "conditional_increment_norm": {
    "n": 36,
    "mean": 0.013462586470711383,
    "median": 0.009283383334093617,
    "min": 0.00017750818188650452,
    "max": 0.04509200842881119
  },
  "head_fraction_of_increment_energy": {
    "n": 36,
    "mean": 0.09284126294963896,
    "median": 0.08748141379650251,
    "min": 0.03497222015089135,
    "max": 0.28528441201203614
  },
  "update_GN_energy_fraction": {
    "n": 36,
    "mean": 0.8373834633087665,
    "median": 0.8823592745481299,
    "min": 0.4656980076341967,
    "max": 0.9842270641740865
  },
  "tail_GN_energy_fraction": {
    "n": 36,
    "mean": 0.16261653669123644,
    "median": 0.11764072545187663,
    "min": 0.01577293582592636,
    "max": 0.5343019923658051
  },
  "cross_GN_energy_fraction": {
    "n": 36,
    "mean": -2.9631555498062686e-15,
    "median": -2.1817786971339736e-15,
    "min": -1.3000686172317955e-14,
    "max": 1.8927768882101662e-16
  },
  "trace_top16": {
    "n": 36,
    "mean": 0.8178525396665473,
    "median": 0.8167230181866232,
    "min": 0.7799236691989379,
    "max": 0.8622233689898874
  }
}
```

### 4–5：跨轨迹边际效果与整步绝对效果

```json
{
  "T_old": {
    "marginal_with_rest": {
      "n": 36,
      "mean": -0.012989810021180246,
      "median": -0.0027766190469264984,
      "min": -0.10925070196390152,
      "max": 0.10476554930210114
    },
    "absolute_keep": {
      "n": 36,
      "mean": -0.028707466502156522,
      "median": -0.007320558652281761,
      "min": -0.2771402783691883,
      "max": 0.22774279117584229
    },
    "absolute_remove": {
      "n": 36,
      "mean": -0.015717656480976276,
      "median": -0.006392054259777069,
      "min": -0.23783227801322937,
      "max": 0.2060137540102005
    },
    "keep_absolute_worse_starts": 15,
    "remove_absolute_worse_starts": 17
  },
  "T_new": {
    "marginal_with_rest": {
      "n": 36,
      "mean": -0.01616907350658824,
      "median": -0.004238840192556381,
      "min": -0.22004137933254242,
      "max": 0.0654730498790741
    },
    "absolute_keep": {
      "n": 36,
      "mean": -0.0007011710975639573,
      "median": 0.011603854596614838,
      "min": -0.3003333956003189,
      "max": 0.3395969569683075
    },
    "absolute_remove": {
      "n": 36,
      "mean": 0.015467902409024662,
      "median": 0.019575566053390503,
      "min": -0.3025498315691948,
      "max": 0.3807690143585205
    },
    "keep_absolute_worse_starts": 21,
    "remove_absolute_worse_starts": 21
  }
}
```

### 5：空间内外交互

```json
{
  "T_old": {
    "value": {
      "n": 36,
      "mean": -0.00048568516245318786,
      "median": 1.6763806343078613e-06,
      "min": -0.01571112871170044,
      "max": 0.0055541349574923515
    },
    "absolute_size": {
      "n": 36,
      "mean": 0.001504262469501959,
      "median": 0.0005445182323455811,
      "min": 1.4454126358032227e-06,
      "max": 0.01571112871170044
    },
    "raw_sign_reversals_with_vs_without": 0,
    "warning": "Raw sign reversals descriptive; see numerical bands and trajectory intervals for discernibility"
  },
  "T_new": {
    "value": {
      "n": 36,
      "mean": -2.246553545446437e-05,
      "median": 5.61736524105072e-05,
      "min": -0.012592077255249023,
      "max": 0.004729963839054108
    },
    "absolute_size": {
      "n": 36,
      "mean": 0.0012403517473702091,
      "median": 0.0003540441393852234,
      "min": 2.428889274597168e-06,
      "max": 0.012592077255249023
    },
    "raw_sign_reversals_with_vs_without": 0,
    "warning": "Raw sign reversals descriptive; see numerical bands and trajectory intervals for discernibility"
  }
}
```

### 6：D选择后各方法在T上的共同增益与新旧代价

```json
{
  "GN": {
    "joint_gain": {
      "n": 36,
      "mean": 0.004075484729481913,
      "median": 0.0,
      "min": -0.034794725477695465,
      "max": 0.08511929959058762
    },
    "old_cost": {
      "n": 36,
      "mean": -0.005117867038481765,
      "median": 0.0,
      "min": -0.10476554930210114,
      "max": 0.023718997836112976
    },
    "new_cost": {
      "n": 36,
      "mean": -0.00303310242048206,
      "median": 0.0,
      "min": -0.0654730498790741,
      "max": 0.045870453119277954
    }
  },
  "C1_0": {
    "joint_gain": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "old_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "new_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    }
  },
  "C1_1": {
    "joint_gain": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "old_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "new_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    }
  },
  "C1_2": {
    "joint_gain": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "old_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "new_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    }
  },
  "C2_0": {
    "joint_gain": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "old_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "new_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    }
  },
  "C2_1": {
    "joint_gain": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "old_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    },
    "new_cost": {
      "n": 36,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0
    }
  },
  "C2_2": {
    "joint_gain": {
      "n": 36,
      "mean": 1.2475169367260403e-05,
      "median": 0.0,
      "min": 0.0,
      "max": 0.0004491060972213745
    },
    "old_cost": {
      "n": 36,
      "mean": -1.2099742889404297e-05,
      "median": 0.0,
      "min": -0.0004355907440185547,
      "max": 0.0
    },
    "new_cost": {
      "n": 36,
      "mean": -1.285059584511651e-05,
      "median": 0.0,
      "min": -0.00046262145042419434,
      "max": 0.0
    }
  },
  "C3": {
    "joint_gain": {
      "n": 36,
      "mean": 0.00021591803265942467,
      "median": 0.0,
      "min": -0.007944993674755096,
      "max": 0.007630351930856705
    },
    "old_cost": {
      "n": 36,
      "mean": -0.0003383962644471063,
      "median": 0.0,
      "min": -0.007477506995201111,
      "max": 0.005170315504074097
    },
    "new_cost": {
      "n": 36,
      "mean": -9.34398008717431e-05,
      "median": 0.0,
      "min": -0.007783196866512299,
      "max": 0.010719671845436096
    }
  }
}
```

### 7：旁观旧任务逐起点结果

```json
[
  {
    "start_id": "s45_t8_e33_b285",
    "role": "T_monitor_6",
    "candidate": "GN_REMOVE",
    "vs_KEEP": -0.0005564689636230469,
    "vs_BASE": 0.1571650505065918
  },
  {
    "start_id": "s45_t8_e48_b91",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": 0.2201451063156128
  },
  {
    "start_id": "s45_t8_e63_b85",
    "role": "T_monitor_6",
    "candidate": "GN_HALF",
    "vs_KEEP": 0.03384518623352051,
    "vs_BASE": 0.06277275085449219
  },
  {
    "start_id": "s45_t8_e74_b56",
    "role": "T_monitor_6",
    "candidate": "GN_HALF",
    "vs_KEEP": 0.002835869789123535,
    "vs_BASE": -0.09105336666107178
  },
  {
    "start_id": "s45_t8_e85_b93",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": -0.23213481903076172
  },
  {
    "start_id": "s45_t8_e99_b94",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": 0.008992552757263184
  },
  {
    "start_id": "s46_t8_e33_b148",
    "role": "T_monitor_6",
    "candidate": "GN_HALF",
    "vs_KEEP": 0.01792389154434204,
    "vs_BASE": -0.010330557823181152
  },
  {
    "start_id": "s46_t8_e48_b302",
    "role": "T_monitor_6",
    "candidate": "GN_REMOVE",
    "vs_KEEP": -0.059114813804626465,
    "vs_BASE": 0.47609853744506836
  },
  {
    "start_id": "s46_t8_e62_b105",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": -0.2747976779937744
  },
  {
    "start_id": "s46_t8_e75_b271",
    "role": "T_monitor_6",
    "candidate": "GN_REMOVE",
    "vs_KEEP": -0.0010204315185546875,
    "vs_BASE": 0.00946807861328125
  },
  {
    "start_id": "s46_t8_e86_b69",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": 0.028449177742004395
  },
  {
    "start_id": "s46_t8_e98_b91",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": 0.0022203922271728516
  },
  {
    "start_id": "s47_t8_e30_b36",
    "role": "T_monitor_6",
    "candidate": "GN_REMOVE",
    "vs_KEEP": -0.018701672554016113,
    "vs_BASE": -0.2947194576263428
  },
  {
    "start_id": "s47_t8_e47_b187",
    "role": "T_monitor_6",
    "candidate": "GN_REMOVE",
    "vs_KEEP": -0.06311392784118652,
    "vs_BASE": -0.10251355171203613
  },
  {
    "start_id": "s47_t8_e64_b87",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": -0.4051220417022705
  },
  {
    "start_id": "s47_t8_e73_b80",
    "role": "T_monitor_6",
    "candidate": "GN_HALF",
    "vs_KEEP": 0.0006897449493408203,
    "vs_BASE": 0.008414983749389648
  },
  {
    "start_id": "s47_t8_e87_b70",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": -0.21224570274353027
  },
  {
    "start_id": "s47_t8_e97_b140",
    "role": "T_monitor_6",
    "candidate": "KEEP",
    "vs_KEEP": 0.0,
    "vs_BASE": 0.08798563480377197
  }
]
```

### 8：固定6点NLL与行为配对结果

```json
[]
```

7：夹爪指令变化窗口和时序分层的尾部损伤见 STATE_AND_TRAJECTORY_STRATA.csv 与 DESCRIPTIVE_ANSWERS.json，不以全局均值代替。9：成本含候选、GN、D/T及KL，T前向与KL合并计时，未单独分离；对照范数补录的影子状态恢复时间单列，不能隐去。10：以上结论仅限固定开发载体，不自动触发续训。
