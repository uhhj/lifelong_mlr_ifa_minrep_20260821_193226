# LIBERO-OBJECT temporal functional-interference diagnosis

## Identity

- Verdict: `TEMPORAL_FUNCTIONAL_INTERFERENCE_SUPPORTED`
- Baseline parent SHA: `7b937ca247b153f5ddd16385746c7709a1a2d376`
- Diagnosis branch: `research/temporal-interference-diagnosis`
- Diagnosis final SHA: the branch HEAD containing this report (reported in the Git handoff; a commit cannot contain its own SHA)
- Server GPU: NVIDIA GeForce RTX 4090, 24,564 MiB; driver 535.230.02
- Environment: Python 3.8.20, PyTorch 1.13.0+cu116, CUDA runtime 11.6, MuJoCo 3.2.3, robosuite 1.4.0, transformers 4.26.1

This is an offline, read-only analysis. It does not modify MLR training, IFA, replay, scheduling, evaluation, or any checkpoint. The policy was put in `eval()` mode and gradients were obtained only with `torch.autograd.grad` from the existing BC objective. The unused GPT-2 token-embedding parameter was represented by its actual zero gradient; no model path was changed.

## Protocol

- Primary method: MLR
- Seeds: 42, 43, 44
- Natural pairs: 6-7, 6-8, 7-8, 6-9, 7-9, 8-9
- Progress bins: five normalized trajectory-progress bins
- Formal sample size: 32 fixed records per seed/task/bin (160 task-level records)
- Gradient batch size: 8
- Groups: all trainable parameters, `temporal_transformer`, and `policy_head`
- Descriptive signs: positive >= +0.05, negative <= -0.05, otherwise near zero
- Replication: mixed-sign in at least 2/3 seeds and diagonal range >= 0.20 in at least two seeds
- Secondary comparison: MLR+IFA for 6-7, 6-8, 7-8, and 7-9

The task-level gradient is the weighted aggregate of the same five fixed bin gradients, so the task-level and temporal comparisons use exactly the same records.

## Data alignment

Action windows were normalized to float32 C-contiguous arrays and aligned by exact bytes to the robomimic 0.2.0 `SequenceDataset` index. Ambiguous matches were excluded.

| Seed | Task | Feature records | Matched | Ambiguous | Unmatched | Coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 6 | 1562 | 1558 | 4 | 0 | 99.744% |
| 42 | 7 | 1480 | 1480 | 0 | 0 | 100.000% |
| 42 | 8 | 1560 | 1560 | 0 | 0 | 100.000% |
| 42 | 9 | 1319 | 1319 | 0 | 0 | 100.000% |
| 43 | 6 | 1562 | 1558 | 4 | 0 | 99.744% |
| 43 | 7 | 1480 | 1480 | 0 | 0 | 100.000% |
| 43 | 8 | 1560 | 1560 | 0 | 0 | 100.000% |
| 43 | 9 | 1319 | 1319 | 0 | 0 | 100.000% |
| 44 | 6 | 1562 | 1558 | 4 | 0 | 99.744% |
| 44 | 7 | 1480 | 1480 | 0 | 0 | 100.000% |
| 44 | 8 | 1560 | 1560 | 0 | 0 | 100.000% |
| 44 | 9 | 1319 | 1319 | 0 | 0 | 100.000% |

The formal feature caches under `features/object_seed42`, `object_seed43`, and `object_seed44` were kept separate and used directly. No latent was re-extracted.

## Checkpoints

For new task 7, 8, or 9, the analysis loads the preceding MLR stage 6, 7, or 8 checkpoint respectively:

```text
runs/object_seed{seed}_mlr/run_001/mlrifa_6_model_ep95.pth
runs/object_seed{seed}_mlr/run_001/mlrifa_7_model_ep95.pth
runs/object_seed{seed}_mlr/run_001/mlrifa_8_model_ep95.pth
```

`ep95` is a late-stage pre-transition proxy, not the exact in-memory ep99 state used by the original continuous training loop. The conclusions therefore concern functional-interference geometry observed near the stage boundary, not the exact causal gradient of the first next-stage update.

## Main scientific table: MLR all-trainable gradients

| Seed | Pair | Task cosine | Bin 0 | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Min | Max | Range | Mixed sign |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 6-7 | 0.063 | 0.332 | -0.126 | -0.090 | 0.098 | 0.217 | -0.126 | 0.332 | 0.458 | yes |
| 42 | 6-8 | 0.033 | 0.463 | 0.632 | -0.297 | 0.630 | -0.307 | -0.307 | 0.632 | 0.939 | yes |
| 42 | 7-8 | -0.063 | 0.415 | -0.093 | -0.005 | 0.581 | -0.505 | -0.505 | 0.581 | 1.086 | yes |
| 42 | 6-9 | 0.118 | 0.087 | 0.051 | 0.113 | 0.671 | 0.203 | 0.051 | 0.671 | 0.620 | no |
| 42 | 7-9 | 0.116 | 0.204 | -0.051 | 0.083 | 0.554 | 0.234 | -0.051 | 0.554 | 0.605 | yes |
| 42 | 8-9 | 0.086 | 0.149 | -0.225 | 0.088 | 0.620 | 0.345 | -0.225 | 0.620 | 0.846 | yes |
| 43 | 6-7 | 0.044 | -0.148 | -0.016 | 0.140 | 0.111 | 0.161 | -0.148 | 0.161 | 0.309 | yes |
| 43 | 6-8 | 0.274 | -0.150 | 0.322 | 0.053 | 0.276 | 0.317 | -0.150 | 0.322 | 0.473 | yes |
| 43 | 7-8 | 0.155 | 0.246 | 0.148 | -0.211 | 0.557 | 0.180 | -0.211 | 0.557 | 0.769 | yes |
| 43 | 6-9 | 0.067 | 0.005 | -0.081 | 0.737 | 0.145 | -0.093 | -0.093 | 0.737 | 0.830 | yes |
| 43 | 7-9 | 0.085 | -0.051 | 0.042 | 0.783 | 0.092 | -0.137 | -0.137 | 0.783 | 0.921 | yes |
| 43 | 8-9 | 0.053 | 0.007 | -0.161 | 0.734 | 0.088 | -0.017 | -0.161 | 0.734 | 0.895 | yes |
| 44 | 6-7 | -0.065 | 0.031 | -0.112 | -0.419 | -0.036 | -0.057 | -0.419 | 0.031 | 0.450 | no |
| 44 | 6-8 | 0.179 | 0.422 | 0.027 | 0.032 | 0.437 | -0.038 | -0.038 | 0.437 | 0.475 | no |
| 44 | 7-8 | 0.139 | 0.301 | 0.196 | 0.068 | 0.443 | -0.243 | -0.243 | 0.443 | 0.686 | yes |
| 44 | 6-9 | 0.172 | 0.083 | 0.145 | 0.159 | 0.413 | 0.212 | 0.083 | 0.413 | 0.330 | no |
| 44 | 7-9 | 0.191 | 0.025 | 0.443 | 0.003 | 0.247 | 0.326 | 0.003 | 0.443 | 0.440 | no |
| 44 | 8-9 | 0.156 | -0.049 | 0.055 | -0.189 | 0.842 | 0.243 | -0.189 | 0.842 | 1.031 | yes |

## Three-seed aggregation

Values are mean +/- sample standard deviation across seeds. Replication is evaluated per the fixed descriptive rule above.

| Pair | Task cosine | Diagonal mean | Mean diagonal range | Mixed seeds | Replicated |
| --- | ---: | ---: | ---: | ---: | --- |
| 6-7 | 0.014 +/- 0.069 | 0.006 +/- 0.109 | 0.406 | 2/3 | yes |
| 6-8 | 0.162 +/- 0.121 | 0.188 +/- 0.032 | 0.629 | 2/3 | yes |
| 7-8 | 0.077 +/- 0.122 | 0.139 +/- 0.054 | 0.847 | 3/3 | yes |
| 6-9 | 0.119 +/- 0.053 | 0.190 +/- 0.043 | 0.593 | 1/3 | no |
| 7-9 | 0.131 +/- 0.055 | 0.186 +/- 0.035 | 0.655 | 2/3 | yes |
| 8-9 | 0.098 +/- 0.053 | 0.169 +/- 0.034 | 0.924 | 3/3 | yes |

Five of six natural MLR pairs meet the replication criterion. Pair 7-8 has an aggregate cosine of -0.063 in seed42 while its local bins span -0.505 to +0.581; this is a direct example of a task-level relation concealing both transfer and interference. Pair 8-9 is mixed in all three seeds with a mean diagonal range of 0.924.

## Module comparison

| Pair | Transformer diagonal mean | Transformer replicated | Head diagonal mean | Head replicated |
| --- | ---: | --- | ---: | --- |
| 6-7 | -0.007 | no | 0.006 | yes |
| 6-8 | 0.158 | no | 0.189 | yes |
| 7-8 | 0.049 | yes | 0.140 | yes |
| 6-9 | 0.043 | yes | 0.191 | no |
| 7-9 | 0.024 | no | 0.188 | yes |
| 8-9 | 0.072 | no | 0.169 | yes |

The policy-head gradient norm is 5.63-55.25 times the transformer norm at task level (median 11.62), so the all-trainable cosine is dominated by the head. The transformer is not homogeneous: 7-8 replicates mixed sign in both modules, while 6-9 replicates mixed sign only in the transformer. Thus H3 does not reduce to a single universal layer, although the dominant measured conflict is in the policy head.

## Action-conditioned proxy and secondary IFA check

The 14-dimensional action-centroid descriptor is included for every 5x5 entry. Across the 90 MLR diagonal entries, its cosine lies in 0.876-1.000 and has Pearson correlation 0.152 with gradient cosine. It is therefore a weak explanatory proxy here and is not treated as a ground-truth manipulation phase.

The MLR+IFA stage6 task6-7 task-level and binwise results match MLR exactly for all three seeds, as expected before IFA can affect a later stage. The targeted later-stage IFA results still contain heterogeneous matrices, but they are secondary and do not determine the verdict. No IFA parameter or pair selection was changed.

## Interpretation

Observed:

- Five of six natural MLR pairs replicate temporal mixed sign by the preregistered descriptive rule; 7-8 and 8-9 do so in all three seeds.
- Task-level cosine can be near zero or have a single sign while individual same-progress bins show large positive and negative gradients.
- Head gradients dominate the concatenated parameter geometry, while transformer-only heterogeneity is also replicated for 7-8 and 6-9.
- Alignment coverage is 99.744-100%, with no unmatched feature records.

Inferred:

- A single task-pair compatibility value is too coarse for these checkpoints and selected latent/action windows.
- Local functional transfer and interference coexist over normalized trajectory progress.
- Any later functional gating study should measure the policy head explicitly and should not assume that decoder and head relations are interchangeable.

Not established:

- Normalized progress is not a semantic reach/grasp/transport/place label.
- This diagnosis does not show that Robot-IGFA, gradient projection, or any other intervention improves FWT, NBT, or AUC.
- The ep95 proxy does not establish exact first-update geometry at the ep99 stage boundary.
- The shared task6 adaptation bottleneck is orthogonal to this diagnostic objective.

## Verdict

`TEMPORAL_FUNCTIONAL_INTERFERENCE_SUPPORTED`

H1 is supported: task-level aggregation hides substantial local variation. H2 is supported: replicated mixed-sign functional relations occur in five natural pairs. H3 indicates dominant policy-head geometry plus pair-specific transformer heterogeneity. The temporal/behavior-conditioned Robot-IGFA premise has empirical support sufficient to continue research, but the next experiment must first test vanilla task-level functional IGFA; no performance benefit is established here.

## Exact successful commands

```bash
cd /data/lifelong_mlr_ifa_minrep
source /miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/mlr_ifa_minrep
export PYTHONPATH="$PWD"
export LIBERO_CONFIG_PATH="$PWD/data/libero_config"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1

pytest -q tests/test_temporal_interference.py
bash scripts/run_temporal_interference_object.sh smoke
bash scripts/run_temporal_interference_object.sh formal
bash scripts/run_temporal_interference_object.sh ifa
python scripts/summarize_temporal_interference.py \
  --input-dir reports/temporal_interference_diagnosis/raw \
  --output-dir reports/temporal_interference_diagnosis
```

The successful smoke outputs were inspected and then deleted before the formal run. Raw 5x5 values, task-level gradients, three-seed summaries, necessary logs, and figures are retained in this report directory.
