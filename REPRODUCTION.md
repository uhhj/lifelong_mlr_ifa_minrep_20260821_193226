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
- New private repository: <https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226>

## Environment

- Server OS: Ubuntu 20.04.4 LTS, Linux 5.15.0-139-generic
- GPU: NVIDIA GeForce RTX 4080 SUPER, single GPU, device index 0
- NVIDIA driver: 535.230.02 (`nvidia-smi` host CUDA compatibility 12.2)
- Python: 3.8.20 in `/data/conda_envs/mlr_ifa_minrep`
- PyTorch: 1.13.0+cu116
- Bundled PyTorch CUDA runtime: 11.6; `torch.cuda.is_available()` is `True`; CUDA matrix multiplication passed
- MuJoCo: 3.2.3
- Headless rendering: EGL environment construction, reset, offscreen render `(128, 128, 3)`, and close all passed
- Transformers / robosuite / robomimic: 4.26.1 / 1.4.0 / 0.2.0
- `pip check`: no broken requirements

## Data and checkpoint

- Dataset: `/data/lifelong_mlr_ifa_minrep/data/libero_object`
- Dataset source: the official LIBERO Hugging Face fallback used by the upstream downloader, `yifengzhu-hf/LIBERO-datasets`, restricted to `libero_object/*`
- Dataset contents: 10 non-empty demonstration HDF5 files, each with 50 demos; all files open successfully with h5py
- Task-6 loader smoke: 10 demonstrations, 1,562 sequences, action dimension 7, both 128x128 RGB observations loaded successfully
- Expected checkpoint: `/data/lifelong_mlr_ifa_minrep/checkpoints/object_multitask_model_ep99.pth`
- Checkpoint source: upstream README “Download pretrained models” Google Drive folder

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
| IFA lambda | 0.1 | hard-coded 0.1 | `ifa_weight=0.1` |
| IFA distance | angular | `torch.acos` angular distance | angular |
| pair selection | language + agent-view top 50% | independent top-50% sets, then intersection | upstream behavior |
| checkpoint selection | best current-task checkpoint | epochs 70–95 every 5 epochs | best current-task success; latest epoch on ties |
| replay size/rate | about 5 demos/task | `past_data_rate=0.5` | 0.5 |

The release argument `gamma=0.31` multiplies the angular distance between task-language references in the adaptive margin. It is therefore the released LIBERO-OBJECT value for the paper's angular scaling factor `alpha` (approximately 0.3), and is unrelated to the LoRA `policy.alpha=8` setting.

Checkpoint selection follows the author's public clarification in [upstream issue 2](https://github.com/yfqi/lifelong_mlr_ifa/issues/2): select the best checkpoint for the current task. The release saves and evaluates epochs 70, 75, 80, 85, 90, and 95. The selected epoch's scores on all seen tasks form that stage's row of `r[i,j]`.

## Method sanity audit

- MLR feature files contain `feature` and `actions`. `feature` is the frozen encoder/FiLM multimodal latent, not raw RGB.
- For a new task, all current feature files are concatenated with replay feature files and shuffled. A 0.5 random subset is copied into the task-specific replay buffer after each stage.
- `BriefPolicy` freezes the inherited pretrained network before constructing a new temporal transformer and policy head. Thus the vision/language/state encoders and modulation layers are frozen, while the temporal decoder and policy head are trainable.
- IFA uses the current-task global temporal latent, its own language latent, old-task language latents, angular distance, an adaptive language-reference margin, and pairs in the intersection of the top-50% language and agent-view similarity sets.
- Total loss is `L_BC + ifa_weight * L_IFA`; the MLR launcher sets `ifa_enabled=false` without disabling replay.

## Commands

Successful preparation commands included:

```bash
cd /data
git clone https://github.com/yfqi/lifelong_mlr_ifa.git lifelong_mlr_ifa_minrep
cd /data/lifelong_mlr_ifa_minrep
git remote rename origin upstream
git remote add origin https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git
conda create -p /data/conda_envs/mlr_ifa_minrep python=3.8 -y
conda activate /data/conda_envs/mlr_ifa_minrep
python -m pip install --upgrade pip setuptools wheel
conda install -y cmake
CMAKE_POLICY_VERSION_MINIMUM=3.5 python -m pip install egl-probe==1.0.2
PIP_INDEX_URL=https://pypi.org/simple python -m pip install -r requirements.txt
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    allow_patterns=["libero_object/*"],
    local_dir="/data/dataset_staging",
)
PY
export PYTHONPATH=/data/lifelong_mlr_ifa_minrep
export LIBERO_CONFIG_PATH=/data/lifelong_mlr_ifa_minrep/data/libero_config
python -m pip check
source scripts/_common.sh
python -c "import torch; x=torch.rand(1024,1024,device='cuda'); print(torch.cuda.get_device_name(0), (x@x).shape)"
```

The upstream root has no `setup.py`, `pyproject.toml`, or `setup.cfg`, so the documented `pip install -e .` cannot apply to this release. The launchers use the verified repository-root `PYTHONPATH` fallback. `CMAKE_POLICY_VERSION_MINIMUM=3.5` is only an installation compatibility setting for the old `egl-probe` build metadata; package versions remain those pinned by upstream.

The following formal commands are prepared but cannot be reported as successful until the official checkpoint is available. Run them from the repository root after activating `/data/conda_envs/mlr_ifa_minrep`:

```bash
bash save_feature_object.sbatch
bash lifelong_adaptation_object.sbatch mlr
bash lifelong_adaptation_object.sbatch mlr_ifa
bash test.sbatch mlr
bash test.sbatch mlr_ifa
python scripts/compute_object_metrics.py \
  --mlr-log runs/object_seed42_mlr/eval_seed42.log \
  --mlr-ifa-log runs/object_seed42_mlr_ifa/eval_seed42.log \
  --output reports/object_seed42_metrics.json
```

## Changes from upstream

- Replaced author-machine paths with project-relative launchers and a local LIBERO path configuration.
- Configured headless EGL and disabled WandB network logging.
- Added separate MLR and MLR+IFA output/buffer paths and an explicit IFA switch/weight.
- Added a 20-trial evaluation launcher over release epochs 70–95, current-task checkpoint selection, and a small implementation of the paper's stated FWT/NBT/AUC formulas.
- Fixed IFA tensor-device placement so its mask and old-task language reference follow the active model device.
- Corrected the launcher import path to the repository root, which exposes the release's `libero.libero` and `libero.lifelong` namespace layout.

## Results

| Method | FWT | NBT | AUC |
| --- | ---: | ---: | ---: |
| MLR | pending | pending | pending |
| MLR + IFA | pending | pending | pending |

## Deviations / current execution blockers

- The official pretrained-model Google Drive folder linked by the upstream README is not retrievable from the server and returns a not-found page through an independent browser path. Exhaustive checks of the upstream repository/issues/releases plus Hugging Face and Zenodo found no author-published mirror. No unofficial checkpoint substitute was used.
- The private GitHub repository was created, but this server has no authenticated GitHub credential, so the local commits are not pushed; `origin` and `upstream` are configured correctly.
- Formal experiments have not been reduced; result fields remain pending rather than reporting smoke or fabricated values.
- The upstream IFA implementation adds one weighted hinge term per selected old-task pair rather than explicitly dividing their sum by `|P|`. This baseline preserves that released behavior and records it instead of silently changing the method.
