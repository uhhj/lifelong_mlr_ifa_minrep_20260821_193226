source /data/mlr_ifa/activate.sh
export ZR_SEED=40 CORESET_SEED=40
export PARENT_RUN=/data/mlr_ifa/experiments/Seeds40_41_three_arm_20261016/seed40/rc
export PROTOTYPE_RUN=/data/mlr_ifa/analyses/Zero_response_residual_prototype_20261017
export ZR_RUN=/root/mlr_ifa/experiments/ZR_coreset_OBJECT_seed40_20261017
export ZR_SOURCE=/root/mlr_ifa/experiment_workspaces/ZR_coreset_seeds40_41_20261017/seed40/source
export CORESET_OUT="$PARENT_RUN"
export PYTHONPATH="$ZR_SOURCE:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ZR_SOURCE"
