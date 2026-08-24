# Feature seed-dependency audit

## Conclusion

The latent feature cache is seed-dependent and must be regenerated for seeds 43 and 44. Seed42 features are not copied or symlinked as substitutes.

## Evidence

The saved representation is produced from fixed demonstrations and a frozen official checkpoint, but the extraction path is not deterministic across experiment seeds:

1. `save_feature.py` calls `control_seed(cfg.seed)` and uses a `RandomSampler`.
2. `learn_one_task(..., save_feature=True)` explicitly calls `policy.train()`.
3. The OBJECT feature launcher sets `aug=true`; the training config has `train.use_augmentation=true`.
4. `BasePolicy.compute_loss(..., save_feature=True)` calls `preprocess_input`, which applies the configured color, affine/translation, and erase augmentation group in training mode.
5. Augmentation modules consume NumPy and Torch randomness. The filename index follows the randomized dataloader order.

Consequently, the frozen encoder and FiLM modules see seed-specific augmented observations, and both feature values and file-to-trajectory ordering can change. Actions originate from the same demonstrations but are stored alongside the seed-specific ordering.

## Cache policy

- Seed42: retain `features/object_seed42/` (5,921 files).
- Seed43: generate `features/object_seed43/` formally.
- Seed44: generate `features/object_seed44/` formally.
- Do not duplicate a common cache and do not regenerate seed42.

All three caches must contain task directories 6–9 with counts 1,562 / 1,480 / 1,560 / 1,319, finite tensors, and successful reloads.
