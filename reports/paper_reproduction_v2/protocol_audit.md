# Paper-level reproduction V2 protocol audit

## Identity and scope

- Starting baseline: `7b937ca247b153f5ddd16385746c7709a1a2d376`
- Official release snapshot: `a721807f2f7fa7f9833dcc85c592d78e9accd00b`
- Benchmark order: official `LIBERO_OBJECT` order `[0,1,2,3,4,5,6,7,8,9]`.
- Pretraining tasks are 0–5; lifelong stages are task IDs 6–9. The dataset files contain 50 demonstrations, while `data.demo_limit=10` makes the release loader use 10 demonstrations per lifelong task.

| Task | Task name | Dataset path | Lifelong stage |
|---:|---|---|---:|
| 0 | pick up the alphabet soup and place it in the basket | `data/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5` | pretrain |
| 1 | pick up the cream cheese and place it in the basket | `data/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5` | pretrain |
| 2 | pick up the salad dressing and place it in the basket | `data/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5` | pretrain |
| 3 | pick up the bbq sauce and place it in the basket | `data/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5` | pretrain |
| 4 | pick up the ketchup and place it in the basket | `data/libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5` | pretrain |
| 5 | pick up the tomato sauce and place it in the basket | `data/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5` | pretrain |
| 6 | pick up the butter and place it in the basket | `data/libero_object/pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5` | 1 |
| 7 | pick up the milk and place it in the basket | `data/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5` | 2 |
| 8 | pick up the chocolate pudding and place it in the basket | `data/libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5` | 3 |
| 9 | pick up the orange juice and place it in the basket | `data/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5` | 4 |

## Paper, release, and current runner

| Item | Paper | Official release | Starting baseline runner | V2 diagnosis |
|---|---|---|---|---|
| Epochs | 100 | 100 | 100 | match |
| Batch size | 10 | `train/adaption.yaml`: 18 | explicit 10 | paper and release differ |
| Optimizer | AdamW | AdamW | AdamW | match |
| Initial LR | 1e-4 | 1e-4 | 1e-4 | configured value matches |
| Betas / eps / weight decay | not fully specified | 0.9/0.999, 1e-8, 0.1 | release values | retained |
| Scheduler | not specified at implementation-unit level | linear warmup 500 and decay, total=`epochs*loader batches`; stepped once per epoch | same | step-unit mismatch inside release |
| Data shuffle | — | seeded `DistributedSampler`, epoch reseeding | same | retained |
| Replay | latent feature + action, rate 0.5 | deterministic seed-local shuffle/copy | same | retained |
| IFA | language + agent top-50% intersection, angular adaptive margin | lambda 0.1, release OBJECT gamma 0.31 | same; pairs 7→6, 8→7, 9→none | unchanged |

## Evaluation

- Each reported cell uses exactly 20 rollouts.
- The evaluator selects the first 20 saved task init states in two groups of 10. It also calls `env.seed(seed + group_index)` before reset and again after reset.
- Success is the environment `done` signal accumulated over the rollout; failed rollouts are not dropped.
- Task6–9 are evaluated in ascending task order. Current-task checkpoint selection uses epochs 70/75/80/85/90/95, highest success, tie to later epoch. Old tasks use that selected stage checkpoint.
- Checkpoint loading strips the DDP `module.` prefix and loads into `CompletePolicy` with `strict=False`, as in the starting baseline.
- Evaluation is not deterministic at the action level: the GMM policy action path remains stochastic. Global Python/NumPy/Torch/CUDA RNGs are reset to the seed before each stage.

## Source-supported issue

The release constructs a scheduler whose units are optimizer steps (`total_steps = epochs × batches`, warmup 500), but calls `scheduler.step()` only after epochs 1–99. This prevents the legacy path from completing a 500-step warmup. V2 exposes the legacy behavior unchanged and a unit-consistent batch-step diagnosis; it does not change IFA or choose a protocol by outcome.
