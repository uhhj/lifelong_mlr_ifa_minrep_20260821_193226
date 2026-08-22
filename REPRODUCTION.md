# LIBERO-OBJECT minimal reproduction

## Scope

- Benchmark: `LIBERO-OBJECT`
- Seed: `42`
- Lifelong tasks: official task indices 6–9 (`start_task=6`, `end_task=10`)
- Methods: MLR and MLR + IFA
- Initialization: official LIBERO-OBJECT multitask checkpoint

This is a minimal single-seed reproduction, not the full three-seed reproduction.

## Source

- Upstream: <https://github.com/yfqi/lifelong_mlr_ifa>
- `UPSTREAM_SHA`: `a721807f2f7fa7f9833dcc85c592d78e9accd00b`
- Private baseline repository: <https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226>

The complete upstream Git history is preserved. The server remotes are `origin` for the private baseline repository and `upstream` for the author repository.

## Environment

- Server OS: Ubuntu 20.04.4 LTS, Linux 5.15.0-139-generic
- Python: 3.8.20 in `/data/conda_envs/mlr_ifa_minrep`
- PyTorch: 1.13.0+cu116; bundled CUDA runtime 11.6
- NVIDIA driver: 535.230.02; host CUDA compatibility 12.2
- MuJoCo: 3.2.3
- Transformers / robosuite / robomimic: 4.26.1 / 1.4.0 / 0.2.0
- Headless rendering: EGL construction, reset, 128x128 offscreen rendering, and close passed
- `pip check`: no broken requirements

Feature generation, both lifelong training runs, smoke testing, and MLR evaluation used one NVIDIA GeForce RTX 4080 SUPER (24 GB). The cloud instance rebooted after MLR evaluation and reassigned one NVIDIA GeForce RTX 3090 (24 GB); MLR+IFA evaluation completed on that GPU with the same driver, checkpoint, seed, and evaluation configuration.

## Data and checkpoint

- Dataset: `/data/lifelong_mlr_ifa_minrep/data/libero_object`
- Dataset source: official LIBERO dataset mirror `yifengzhu-hf/LIBERO-datasets`, restricted to `libero_object/*`
- Dataset contents: 10 non-empty HDF5 demonstration files, 50 demos each; all open with h5py
- Checkpoint: `/data/lifelong_mlr_ifa_minrep/checkpoints/object_multitask_model_ep99.pth` (664 MiB)
- Checkpoint source: the user-provided local copy of the official OBJECT checkpoint linked by the upstream README, uploaded to the server without modification
- Checkpoint validation: `torch.load(..., map_location="cpu")` succeeded and exposed `cfg`, `previous_masks`, and `state_dict`

Formal feature generation produced 5,921 finite, reloadable files under `features/object_seed42/`: task 6 = 1,562, task 7 = 1,480, task 8 = 1,560, task 9 = 1,319. Each file stores the encoder/FiLM multimodal latent feature and its action, not raw RGB replay trajectories.

## Paper-to-release configuration

| Item | Paper | Current upstream release | This baseline |
| --- | --- | --- | --- |
| benchmark | LIBERO-OBJECT | `LIBERO_OBJECT` | `LIBERO_OBJECT` |
| seed | 3 seeds | launcher default 42 | 42 |
| pretrained tasks | 6 | official checkpoint | official checkpoint |
| lifelong tasks | 4 | `start_task=6,end_task=10` | tasks 6–9 |
| new-task demos | 10 | `demo_limit=10` | 10 |
| batch size | 10 | adaptation YAML default 18 | 10 |
| epochs | 100 | 100 | 100 |
| optimizer | AdamW | AdamW | AdamW |
| learning rate | 1e-4 | 1e-4 | 1e-4 |
| IFA lambda | 0.1 | loss term | `ifa_weight=0.1` |
| IFA distance | angular | `torch.acos` | angular |
| pair selection | language + agent-view top 50% | independent top-50% sets, then intersection | upstream behavior |
| checkpoint selection | best current-task checkpoint | epochs 70–95 every 5 epochs | best current-task success; latest epoch on ties |
| replay size/rate | about 5 demos/task | `past_data_rate=0.5` | 0.5 |

The release argument `gamma=0.31` multiplies the angular distance between current- and old-task language references in the adaptive margin. It is the released LIBERO-OBJECT value corresponding to the paper's angular scaling factor `alpha` (approximately 0.3), and is unrelated to LoRA `policy.alpha=8`.

Checkpoint selection follows the author's public clarification in [upstream issue 2](https://github.com/yfqi/lifelong_mlr_ifa/issues/2): choose the checkpoint with the best current-task success among epochs 70, 75, 80, 85, 90, and 95. Ties use the later epoch. The selected checkpoint is then evaluated on all earlier tasks to form the row `r[i,j]`. Every reported entry uses 20 trials.

## Method sanity audit

- MLR replay files contain frozen encoder/FiLM multimodal latent representations plus actions. Current-task latent files are concatenated with replay files and shuffled; after each stage, `past_data_rate=0.5` samples are copied into that method's independent buffer.
- `BriefPolicy` freezes the inherited pretrained network before creating a new temporal transformer and policy head. The vision, language, state, and modulation modules are frozen; the temporal decoder and policy head are trainable.
- IFA uses the current-task global temporal latent, its own task-language reference, old-task language references, `torch.acos` angular distances, a language-dependent adaptive margin, and the intersection of the language and agent-view top-50% pair sets.
- Total loss is `L_BC + ifa_weight * L_IFA`. MLR sets `ifa_enabled=false` while retaining MLR replay; MLR+IFA sets `ifa_enabled=true`, `ifa_weight=0.1`, and `gamma=0.31`.
- Formal MLR+IFA logs show selected old-task IDs `[6]` at task 7, `[7]` at task 8, and `[]` at task 9. The empty task-9 intersection is preserved rather than forcing a pair.
- The upstream implementation sums a weighted hinge term for each selected old-task pair instead of explicitly dividing by `|P|`; this baseline preserves that released behavior.

## Execution validation

- A temporary one-task, one-epoch smoke passed latent creation, checkpoint loading, finite GPU forward/backward, checkpoint writing, EGL environment launch, and rollout. Smoke outputs were deleted before formal runs.
- Formal MLR and MLR+IFA each completed four stages x 100 epochs with 400 finite epoch losses.
- Each method produced 24 non-empty, finite, CPU-reloadable checkpoints (tasks 6–9 x epochs 70, 75, 80, 85, 90, 95).
- Independent buffers contain 2,961 files each: `buffers/object_seed42_mlr/` and `buffers/object_seed42_mlr_ifa/`.
- Evaluation logs contain 30 required 20-trial groups per method: 24 current-task checkpoint candidates plus 6 old-task entries at the selected checkpoints.

## Successful commands

Checkpoint upload from the user's machine:

```powershell
scp -P 13240 "C:\Users\Lenovo\Downloads\object_multitask_model_ep99.pth" root@jq1.9gpu.com:/data/lifelong_mlr_ifa_minrep/checkpoints/object_multitask_model_ep99.pth
```

Preparation and formal execution from the server repository root:

```bash
cd /data/lifelong_mlr_ifa_minrep
source /miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/mlr_ifa_minrep
source scripts/_common.sh

python -c 'import torch; torch.load("checkpoints/object_multitask_model_ep99.pth", map_location="cpu"); print(torch.__version__, torch.cuda.is_available())'
PYTHONUNBUFFERED=1 bash save_feature_object.sbatch 2>&1 | tee runs/object_seed42_feature.log
PYTHONUNBUFFERED=1 bash lifelong_adaptation_object.sbatch mlr 2>&1 | tee runs/object_seed42_mlr_train.log
PYTHONUNBUFFERED=1 bash lifelong_adaptation_object.sbatch mlr_ifa 2>&1 | tee runs/object_seed42_mlr_ifa_train.log
PYTHONUNBUFFERED=1 bash test.sbatch mlr 2>&1 | tee runs/object_seed42_mlr_eval_stdout.log
PYTHONUNBUFFERED=1 bash test.sbatch mlr_ifa 2>&1 | tee runs/object_seed42_mlr_ifa_eval_stdout.log
python scripts/compute_object_metrics.py \
  --mlr-log runs/object_seed42_mlr/eval_seed42.log \
  --mlr-ifa-log runs/object_seed42_mlr_ifa/eval_seed42.log \
  --output reports/object_seed42_metrics.json
```

The upstream root has no `setup.py`, `pyproject.toml`, or `setup.cfg`, so its documented `pip install -e .` cannot apply to this release. Launchers use the verified repository-root `PYTHONPATH` fallback. No Slurm service was installed or used.

## Changes from upstream

- Replaced author-machine paths with project-relative launchers and local dataset/checkpoint/output configuration.
- Configured headless EGL and disabled WandB network logging.
- Added separate MLR and MLR+IFA output and replay directories plus an explicit IFA on/off setting and weight.
- Added a 20-trial, two-phase evaluation launcher: evaluate all six current-task checkpoint candidates, select the best, then evaluate only required old-task matrix entries.
- Added a small FWT/NBT/AUC aggregation script using the paper/release definitions.
- Fixed IFA tensor-device placement so the mask and old-task language reference follow the active device.
- Added an opt-in `save_final_checkpoint` setting used only by smoke; formal checkpoint behavior is unchanged.

## Results

| Method | FWT | NBT | AUC | Selected epochs (tasks 6/7/8/9) |
| --- | ---: | ---: | ---: | --- |
| MLR | 76.250 | 11.111 | 70.729 | 85 / 75 / 75 / 80 |
| MLR + IFA | 78.750 | 12.500 | 71.875 | 85 / 80 / 80 / 80 |

The complete `r[i,j]` matrices are in `reports/object_seed42_metrics.json`. This single seed improves FWT and AUC with IFA but has slightly worse NBT because task 6 is forgotten more strongly. No checkpoint, task, rollout, seed, or hyperparameter was selected to make the results resemble the paper table.

## Deviations

- This is one seed (`42`), not the paper's three-seed mean.
- The cloud provider rebooted and reassigned the single GPU between method evaluations; MLR+IFA evaluation ran on an RTX 3090 instead of the RTX 4080 SUPER used for training and MLR evaluation. Both used the same software stack and 24 GB GPU memory.
- The release's top-50% intersection produced no IFA pair at task 9; the official behavior was retained.
- The released IFA loss sums selected-pair hinge terms rather than explicitly normalizing by the number of pairs; it was not rewritten.

