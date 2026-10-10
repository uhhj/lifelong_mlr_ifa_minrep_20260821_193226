# Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment

## Completed OBJECT seed42 full-coverage E1-O

E1-O completed400 epochs,105500 optimizer steps and880 formal episodes; independent
acceptance PASS. Exact per-epoch sample counts were preserved while selected pending
old samples were reordered. Primary AUC62.396 versus E0 77.708 did not meet the
prespecified improvement condition, so no additional seeds were run.

- [Results, all-task matrices and evidence](reports/object_seed42_coverage_order_E1O/README.md)
- [Frozen implementation, tests and protocol](experiments/object_seed42_coverage_order_E1O/README.md)


## Completed OBJECT seed42 uniform-pool E1-U

E1-U completed 400 epochs, 105,500 optimizer steps and 880 formal closed-loop
episodes, with independent acceptance PASS. It retains E1's candidate pools,
refresh schedule and replacement gate, using uniform within-pool probabilities.
Primary AUC was 73.646 versus E1's 77.292 and E0's 77.708; the prespecified
extension condition was not met, so no additional seeds were run.

- [Results, matrices and verified evidence](reports/object_seed42_uniform_pool_E1U/README.md)
- [Frozen implementation, tests and protocol](experiments/object_seed42_uniform_pool_E1U/README.md)
- [Independent acceptance](reports/object_seed42_uniform_pool_E1U/evidence/INDEPENDENT_ACCEPTANCE.json)

E0 and E1 remain unchanged. This single-seed observation does not establish a
stable or causal effect; the report preserves hardware and historical-data gaps.

## Completed OBJECT seed42 protective-selection E1

The independent OBJECT E1 experiment completed 400 training epochs, 105,500
optimizer updates and 880 closed-loop episodes. Its primary epoch99 AUC was
77.292 versus the archived E0 baseline's 77.708 (difference −0.417 percentage
points). The prespecified primary endpoint did not improve, so no additional
seeds or parameter search were started. E0 was not retrained.

- [Results, matrices and evidence](reports/object_seed42_protective_selection_E1/README.md)
- [Isolated implementation and execution notes](experiments/object_seed42_protective_selection_E1/README.md)
- [Independent acceptance](reports/object_seed42_protective_selection_E1/SERVER_INDEPENDENT_AUDIT.json)

This separate OBJECT experiment retains its E0 settings, including `use_state=False`
and `gamma=0.3`. It is distinct from the GOAL state-v2 experiment below and its
read-only paired audit also called E1. Its source is archived in an isolated
directory; the existing GOAL implementation remains unchanged.

## MLR branch: completed GOAL seed42 state-v2 run

This branch archives the verified MLR+IFA run based on upstream commit
`a721807f2f7fa7f9833dcc85c592d78e9accd00b`. It does not merge the separate OBJECT
experiment changes from this repository's `main` branch.

- **Completed:** 4 tasks × 100 epochs, 76,100 optimizer steps, 4 final epoch99 checkpoints, 200 closed-loop episodes.
- **Results:** FWT 78.75%, NBT 1.1111 percentage points, AUC 78.4375%.
- **Modalities:** vision, language and state explicitly enabled; the official checkpoint's original state flag remains unknown.
- [Final report](reports/goal_seed42_state_v2/REPORT.md), [reproduction instructions](REPRODUCTION_GOAL_STATE_V2.md), [acceptance evidence](reports/goal_seed42_state_v2/run/final_acceptance.json).
- Weights, demonstrations and latent tensors remain on the execution server and are not included in this branch.
- [E1 paired audit](reports/goal_seed42_state_v2_paired_audit_E1/README.md): read-only analysis of the same 200 episodes, with 12 comparison summaries, 80 initial-state histories and the signed NBT decomposition. No new training or rollouts.


[![arXiv](https://img.shields.io/badge/arXiv-2603.10929-b31b1b.svg)](https://arxiv.org/abs/2603.10929)

<p align="center">
  <img src="assets/main.png" width="85%">
</p>

Official implementation of the paper  "LIL-MLR-IFA"

Accepted at **CVPR 2026**.

MLR-IFA consists of two main components:

- **Multimodal Latent Replay (MLR)**: stores compact latent features for replay instead of raw demonstrations.
- **Incremental Feature Adjustment (IFA)**: stabilizes lifelong adaptation by regularizing task representations when learning new tasks.

This repository is currently under active cleanup.

## Current Release Status

The current uploaded code mainly includes the **lifelong learning stage** for:

- `LIBERO-GOAL`
- `LIBERO-OBJECT`

Specifically, this version includes scripts for:

1. Saving latent features for accelerating lifelong adaptation.
2. Running lifelong adaptation.
3. Testing/evaluating the lifelong policy.

The multitask pretraining code and full LIBERO-50 support will be uploaded later.

## Pretrained Models

The pretrained multitask checkpoints are available on Google Drive:

[Download pretrained models](https://drive.google.com/drive/folders/1NOiyhkb1Hni_ElhqcN7GkiXnfCqOKiSH)

After downloading, please place the checkpoints under a local directory, for example:

## Environment Setup

We recommend creating a clean conda environment with Python 3.8 and installing dependencies using `pip`.

```bash
conda create -n libero python=3.8
conda activate libero

pip install -r requirements.txt
pip install -e .
```

## Reproduction Instructions

The current release supports the lifelong adaptation stage for `LIBERO-GOAL` and `LIBERO-OBJECT`.

### Reproduce Lifelong Adaptation

Taking `LIBERO-GOAL` as an example, first save replay features:

```bash
./save_feature_goal.sbatch
```

Then run lifelong adaptation:

```bash
./lifelong_adaptation_goal.sbatch
```

Finally evaluate the trained policy:

```bash
./test.sbatch
```

Before running, please modify the file and directory paths in the corresponding `.sbatch` scripts.

The workflow for `LIBERO-OBJECT` is the same. Use the corresponding object scripts, such as:

```bash
./save_feature_object.sbatch
./lifelong_adaptation_object.sbatch
```

### Additional Notes

1. Some scripts may manually add the system path, for example:

```python
sys.path.insert(0, "/work")
```

Please modify this path according to your local project directory before running.

2. The `buffer_dir` is recommended to be used only once for each run. To avoid loading stale replay buffers from previous experiments, please use a new buffer directory when running a new experiment, for example:

```bash
+buffer_dir=./buffer_goal_run1
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{yu2026lifelong,
  title={Lifelong imitation learning with multimodal latent replay and incremental adjustment},
  author={Yu, Fanqi and Tiezzi, Matteo and Apicella, Tommaso and Beyan, Cigdem and Murino, Vittorio},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6740--6749},
  year={2026}
}
```
## TODO

- [ ] Upload multitask pretraining scripts.
- [ ] Upload scripts and configs for LIBERO-50.
