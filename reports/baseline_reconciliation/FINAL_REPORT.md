# LIBERO-OBJECT baseline reconciliation final report

## Verdict

`BASELINE_RECONCILED_WITH_DOCUMENTED_DELTA`

All six formal method-by-seed runs completed at 100 epochs and all evaluations contain exactly 30 success records generated from 20 trials per entry. The pipeline is internally consistent and repeatable. MLR AUC reconciles to the paper scale, while a stable task6 FWT gap and seed-dependent IFA forgetting remain documented numerical deltas.

## Identity and environment

- Starting SHA: `ae2f566697e9e24cc73297ce0451dad5fe7cba98`
- Final SHA: the immutable commit containing this report; obtain exactly with `git rev-parse HEAD` on `research/baseline-reconciliation` (also recorded in the final handoff).
- Branch: `research/baseline-reconciliation`
- Repository: `uhhj/lifelong_mlr_ifa_minrep_20260821_193226`
- Server: Ubuntu 20.04.4, Linux 5.15.0-139; provider-assigned single 24 GB NVIDIA GPU (RTX 4080 SUPER or RTX 3090 across runs), driver 535.230.02.
- Runtime: Python 3.8.20; PyTorch 1.13.0+cu116; Torch CUDA 11.6; cuDNN 8.3.2; MuJoCo 3.2.3; robosuite 1.4.0; robomimic 0.2.0; Transformers 4.26.1.

## Feature seed dependency

Features are seed-dependent. Extraction uses `RandomSampler`, policy train mode, and enabled random image augmentation, so seeds43/44 were regenerated rather than copied from seed42. Each cache contains 5,921 finite, reloadable latent/action records with task counts 1,562 / 1,480 / 1,560 / 1,319.

## Results

Values are percentages; three-seed standard deviations are sample standard deviations.

| Method | Seed | FWT | NBT | AUC |
| --- | ---: | ---: | ---: | ---: |
| MLR | 42 | 76.250 | 11.111 | 70.729 |
| MLR | 43 | 77.500 | -7.222 | 81.562 |
| MLR | 44 | 78.750 | 4.444 | 76.562 |
| MLR+IFA | 42 | 78.750 | 12.500 | 71.875 |
| MLR+IFA | 43 | 77.500 | -3.056 | 79.271 |
| MLR+IFA | 44 | 78.750 | 11.111 | 73.438 |

| Method | Metric | Ours mean ± sample std | Paper | Difference | Standardized descriptive gap |
| --- | --- | ---: | ---: | ---: | ---: |
| MLR | FWT | 77.500 ± 1.250 | 83.3 ± 2.6 | -5.800 | -2.231 |
| MLR | NBT | 2.778 ± 9.279 | 12.3 ± 2.4 | -9.522 | -3.968 |
| MLR | AUC | 76.284 ± 5.422 | 77.6 ± 3.0 | -1.316 | -0.439 |
| MLR+IFA | FWT | 78.333 ± 0.722 | 84.6 ± 1.9 | -6.267 | -3.298 |
| MLR+IFA | NBT | 6.852 ± 8.608 | 11.4 ± 5.6 | -4.548 | -0.812 |
| MLR+IFA | AUC | 74.861 ± 3.898 | 79.4 ± 1.5 | -4.539 | -3.026 |

Mean MLR+IFA minus MLR is Delta FWT=+0.833, Delta NBT=+4.074, and Delta AUC=-1.423. Per-seed deltas are stored in `object_3seed_metrics.json`.

## Success-matrix diagnosis

- Seed42 is the lowest MLR AUC seed, but not an isolated FWT outlier. All three MLR FWT values lie in 76.25–78.75.
- The shared FWT deficit is concentrated in task6: its three-seed FWT mean is 40.0 for both methods, while task7/task8/task9 means are 93–97 / 100 / 76.667.
- MLR AUC is reconciled descriptively: seed43 is above the paper mean, seed44 is close, and the three-seed mean gap is only -1.316.
- AUC reflects both the low task6 starting point and old-task changes. Seed43 MLR shows positive backward transfer; seed42 IFA forgets task6; seed44 IFA loses task7 sharply at stage9.
- The task7 -> task6 IFA pair effect at stage7 is -35/-30/+5 points for seeds42/43/44, so it is not stable across seeds. Seed42's task8 -> task7 pair corresponds to +35 points at stage8.
- Task9 has no IFA pair and identical new-task FWT in both methods for every seed. Nevertheless, inherited model trajectories differ; seed44 task7 falls from 75 at stage8 to 15 at stage9 under IFA.

Main cause classification: **D. mixed / unresolved**. The evidence separates a shared, stable task6/release-protocol FWT gap from seed-specific AUC variance and method-specific forgetting. See `diagnosis.md` and the six CSV matrices for the complete evidence.

## Remaining uncertainty and documented deviations

- All versions pinned by the release match. The author GPU, driver, OS, and cuDNN versions are unpublished.
- The official release applies its warmup scheduler once per epoch even though `total_steps` is expressed in batches. This long effective warmup is preserved and is a plausible, not proven, source of a shared paper/release gap.
- Evaluation preserves release stochastic GMM action sampling over fixed official initial states. Paper seed identities are unpublished; seeds43/44 are local reproducibility seeds, not claimed author seeds.
- IFA mathematics, `lambda_IFA=0.1`, `gamma=0.31`, top-50% intersection, replay, epochs, batch size, LR, and task-pair selection were not changed.
- A transient Hugging Face TLS failure occurred after 28/30 seed44 IFA records. The two missing stage9 old-task records were completed from the existing local CLIP cache with offline mode, using the same checkpoint, seed, initial states, and 20-trial evaluator. Existing records were not rerun or altered.

## Exact successful commands

```bash
cd /data/lifelong_mlr_ifa_minrep
source /miniconda3/etc/profile.d/conda.sh
conda activate mlr_ifa_minrep
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1

SEED=43 bash save_feature_object.sbatch
SEED=44 bash save_feature_object.sbatch

SEED=43 bash lifelong_adaptation_object.sbatch mlr
SEED=43 bash lifelong_adaptation_object.sbatch mlr_ifa
SEED=44 bash lifelong_adaptation_object.sbatch mlr
SEED=44 bash lifelong_adaptation_object.sbatch mlr_ifa

SEED=43 bash test.sbatch mlr
SEED=43 bash test.sbatch mlr_ifa
SEED=44 bash test.sbatch mlr
SEED=44 bash test.sbatch mlr_ifa
```

The last command produced 28 durable records before a remote TLS failure. The exact behavior-preserving completion command for the two missing records was:

```bash
source scripts/_common.sh
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
"$PYTHON" libero/lifelong/test.py \
  seed=44 benchmark_name=LIBERO_OBJECT policy=complete_policy lifelong=basic \
  folder="$DATA_ROOT" bddl_folder="$PROJECT_ROOT/libero/libero/bddl_files" \
  init_states_folder="$PROJECT_ROOT/libero/libero/init_files" pretrain="$CHECKPOINT" \
  eval.n_eval=20 eval.num_procs=20 +start_task=6 +end_task=10 +aug=true \
  +epoch_dir="$PROJECT_ROOT/runs/object_seed44_mlr_ifa/run_001" \
  +test_log_file="$PROJECT_ROOT/runs/object_seed44_mlr_ifa/eval_seed44.log" \
  training=false +eval_stage=9 '+eval_task_ids=[7,8]' '+eval_epochs=[80]' \
  +experiment_root="$PROJECT_ROOT/runs/object_seed44_mlr_ifa_eval_stage9"

python scripts/compute_object_multiseed_metrics.py \
  --seeds 42 43 44 --runs-root runs \
  --output-dir reports/baseline_reconciliation
```

## Next action

Create `research/temporal-interference-diagnosis`, then measure whether task-pair gradient compatibility changes with trajectory / behavior phase without changing baseline optimization. Do not implement Robot-IGFA yet.
