source /data/mlr_ifa/experiment_workspaces/E1U_uniform_pool_20261009/source/protective_e1/env.sh
VIEW=primary
LOG_FILE=/data/mlr_ifa/experiments/E1U_uniform_pool_seed42_20261009/probe_no_rollout.log
ARGS=(seed=42 benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic
 folder=/data/mlr_ifa/datasets bddl_folder="$E1_SOURCE/libero/libero/bddl_files"
 init_states_folder="$E1_SOURCE/libero/libero/init_files"
 pretrain=/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth
 eval.n_eval=20 eval.num_procs=20 +start_task=6 +end_task=10 +aug=true
 +epoch_dir="$E1_OUT/run_001" +test_log_file="$LOG_FILE" training=false use_wandb=false hydra.run.dir="$E1_OUT/hydra_outputs/${VIEW}_$(date +%s%N)" hydra.job.chdir=false)

python -B -m protective_e1.evaluate "${ARGS[@]}" '+eval_epochs=[99]' +experiment_root="$E1_OUT/eval_primary" --cfg job --resolve
