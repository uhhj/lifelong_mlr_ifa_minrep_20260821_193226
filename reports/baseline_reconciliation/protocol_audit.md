# Baseline reconciliation protocol audit

## Scope and source

- Benchmark: `LIBERO_OBJECT`, task order index 0, 10 tasks total.
- Official release lineage: `yfqi/lifelong_mlr_ifa` at `a721807f2f7fa7f9833dcc85c592d78e9accd00b`.
- Pretraining tasks: IDs 0–5. Lifelong stages: IDs 6–9.
- Every lifelong dataset is truncated to the first 10 demonstrations by `demo_limit=10`.

## Task order and data

| Task ID | Task name | Dataset path | Stage |
| ---: | --- | --- | --- |
| 0 | pick up the alphabet soup and place it in the basket | `data/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5` | pretraining |
| 1 | pick up the cream cheese and place it in the basket | `data/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5` | pretraining |
| 2 | pick up the salad dressing and place it in the basket | `data/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5` | pretraining |
| 3 | pick up the BBQ sauce and place it in the basket | `data/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5` | pretraining |
| 4 | pick up the ketchup and place it in the basket | `data/libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5` | pretraining |
| 5 | pick up the tomato sauce and place it in the basket | `data/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5` | pretraining |
| 6 | pick up the butter and place it in the basket | `data/libero_object/pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5` | lifelong 1 |
| 7 | pick up the milk and place it in the basket | `data/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5` | lifelong 2 |
| 8 | pick up the chocolate pudding and place it in the basket | `data/libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5` | lifelong 3 |
| 9 | pick up the orange juice and place it in the basket | `data/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5` | lifelong 4 |

The benchmark task map, BDDL order, HDF5 filenames, launchers, and executed `start_task=6,end_task=10` agree.

## Effective training protocol

| Setting | Effective value | Evidence / note |
| --- | --- | --- |
| epochs | 100, epochs 0–99 | MLR loop and formal logs |
| batch size | 10 | launcher override; release adaptation YAML defaults to 18 |
| optimizer | AdamW | release optimizer config |
| base optimizer LR | 1e-4 | launcher and optimizer config |
| betas / epsilon | (0.9, 0.999) / 1e-8 | optimizer config |
| weight decay | 0.1 | optimizer config |
| scheduler | release custom linear warmup/decay via `LambdaLR` | warmup 500, total steps defined as epochs x batches |
| gradient clipping | 100 | adaptation config |
| seed | 42, 43, or 44 | launcher `SEED`, default 42 |
| workers | 0 | formal launchers |
| dataloader shuffle | enabled | seed-shuffled path list plus `DistributedSampler.set_epoch(epoch)` |
| buffer capacity | 59,660,000 files, effectively non-binding here | release constant |
| buffer sampling | seeded random 50% of each current task's latent sequences | `past_data_rate=0.5`, independent method/seed directories |

The release scheduler is noteworthy but is not changed: `LambdaLR` applies the warmup lambda at construction and is stepped once per epoch, although `total_steps` is expressed in batches. Thus the configured AdamW base LR is 1e-4 while the effective LR follows a long release-defined warmup. New-run logs record the actual LR per epoch. This is official-release behavior and a plausible shared numerical delta from a plain constant-1e-4 reading of the paper.

## IFA protocol

The implementation uses the current-task global temporal latent, own-task and old-task CLIP language references, pairwise language and agent-view cosine similarities, the intersection of their independent top-50% pair sets, angular distances via `torch.acos`, and the adaptive margin `gamma * acos(language_similarity)`. The total loss is `L_BC + 0.1 * L_IFA`; `gamma=0.31`.

Formal pair selections were:

| Seed | Task7 | Task8 | Task9 |
| ---: | --- | --- | --- |
| 42 | old task6 | old task7 | none |
| 43 | old task6 | none | none |
| 44 | old task6 | none | none |

Pair selection is seed-dependent because it is computed from each seed's augmented latent cache. The release sums selected-pair hinge terms. At these OBJECT stages there is at most one selected old-task pair, so sum versus mean has no numerical effect. No pair is forced for task9.

## Seed control

- `control_seed` sets Python `random`, NumPy, CPU Torch, all CUDA generators, and `torch.backends.cudnn.deterministic=True`.
- Model/head initialization, dropout, feature augmentation, feature sampling order, mixed latent path order, and buffer sampling are seed-controlled.
- Feature extraction and training use `num_workers=0`, avoiding untracked worker seeds.
- `DistributedSampler` uses its release default sampler seed and `set_epoch(epoch)`. The underlying file-path list is itself shuffled with the experiment seed, so seed still changes sample-to-index assignment.
- Evaluation resets Python/NumPy/Torch/CUDA seeds to the experiment seed at each stage. GMM actions are stochastic because `get_action` calls `dist.sample()` with `low_eval_noise=false`.
- No Python hash seed is set. No evidence was found that hash iteration order enters dataset, buffer, task, or evaluation ordering.

The original local launchers hard-coded 42. They now accept `SEED=<integer>` while defaulting to 42, so an invocation without `SEED` is behaviorally identical to the completed seed42 baseline.

## Evaluation protocol

- Each reported success entry uses exactly 20 trials. The evaluator internally uses 10 subprocess environments and two batches.
- Initial states are loaded from each benchmark task's official `init_states_file`. Trials use indices 0–19 in order; they are not randomly sampled per seed.
- Before each 10-environment batch, the environment is seeded with `seed` and `seed+1`, reset, seeded again, and then overwritten with the fixed serialized initial states. The seed can still affect environment-side NumPy randomness after state restoration.
- Success is the environment `done`, replaced on every step by the task's BDDL goal-predicate conjunction through `_check_success()`.
- Rollouts stop at success for all environments or at `max_steps=600`. Failed trials are retained.
- Checkpoints 70, 75, 80, 85, 90, and 95 are evaluated on the current task. The highest current-task success is selected; ties use the later epoch. Earlier tasks are then evaluated at that selected checkpoint.
- Evaluation order is stages 6–9, candidate epochs ascending, then selected-checkpoint old tasks in ascending task ID.
- Checkpoint state keys have their DDP `module.` prefix stripped and are loaded with `strict=False`, matching the release architecture split.
- Evaluation actions are stochastic but reproducibly seeded, not deterministic modes. This is a release-level evaluation choice and is preserved.

## Audit conclusion

Task/data/order, training scale, replay, IFA, checkpoint selection, and 20-trial accounting are internally consistent. The two protocol details most relevant to numerical reconciliation are the release-defined long LR warmup and stochastic GMM evaluation on fixed initial states. Neither is modified in this reconciliation.
