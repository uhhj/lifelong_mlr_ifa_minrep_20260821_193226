from libero.lifelong.algos.base import Sequential, get_algo_class, get_algo_list
from libero.lifelong.algos.mlr import MLR
from libero.lifelong.algos.mlr_static_robot_igfa import MLRStaticRobotIGFA
from libero.lifelong.algos.mlr_temporal_robot_igfa import MLRTemporalRobotIGFA
from libero.lifelong.algos.mlr_per_task_temporal_robot_igfa import MLRPerTaskTemporalRobotIGFA
from libero.lifelong.algos.mlr_zero_threshold_directional_guard import (
    MLRZeroThresholdDirectionalGuard,
)
from libero.lifelong.algos.er import ER
from libero.lifelong.algos.agem import AGEM
from libero.lifelong.algos.ewc import EWC
from libero.lifelong.algos.packnet import PackNet
from libero.lifelong.algos.single_task import SingleTask
from libero.lifelong.algos.basic import Basic
from libero.lifelong.algos.multitask_pretraining import multitaskpretraining
