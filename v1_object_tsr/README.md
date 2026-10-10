# V1-OBJECT-TaskSelectiveReplay

This independent experiment preserves the original MLR+IFA implementation and
only changes the historical task source of existing replay slots. The user
approved adaptation to the actual baseline: empty initial replay, historical
task counts 0/1/2/3, original 50% window cache, and 100 signature windows sampled
from each task's planned cache. There is no base-task replay.

Run in the existing migration environment, single GPU:

```bash
source .migration/activate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
python -m torch.distributed.run --nproc_per_node=1 --master_port=29561 -m v1_object_tsr.smoke
python -m v1_object_tsr.evaluate_spawn_v1 --spawn-import-smoke
python -m v1_object_tsr.prepare
bash v1_object_tsr/launch.sh
```

`prepare` refuses to overwrite the protocol and validates smoke source hashes.
The queue uses an exclusive lock. Training resumes only from its own complete
epoch recovery and immutable stage buffer transaction. Completed evaluation
stages are validated and skipped; partially consumed evaluation stages fail
stop for explicit recovery, rather than silently rerunning trials.

The initial authorized run is seed42 only: 400 epochs and 105500 optimizer
updates. It then evaluates V1 and saved baseline checkpoints on the original
200-trial primary matrix each, plus 120 final base-task trials each. Thus 640
rollouts supply paired outcomes and the final ten-task SR without retraining
the baseline. Original AUC/FWT/NBT still use the four incremental tasks.

Reports are under `reports/v1_object_tsr/`; large model and signature files
remain under `runs/v1_object_tsr/`. No E1, P1, G1, QP, new losses, or old queues
are enabled. A positive single-seed screen does not automatically start more
seeds or establish statistical significance.
