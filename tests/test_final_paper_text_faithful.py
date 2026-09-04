import json
from pathlib import Path

from libero.lifelong.algos.mlr import should_save_mlr_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "reports/final_paper_text_faithful/PROTOCOL.json").read_text(encoding="utf-8")
)
TRAIN = (ROOT / "scripts/run_object_final_paper_text_train.sh").read_text(encoding="utf-8")
EVAL = (ROOT / "scripts/eval_object_final_paper_text.sh").read_text(encoding="utf-8")
RUNNER = (ROOT / "scripts/run_final_paper_text_faithful.sh").read_text(encoding="utf-8")
SUMMARY = (ROOT / "scripts/summarize_final_paper_text_faithful.py").read_text(encoding="utf-8")


def test_t01_batch_is_10():
    assert PROTOCOL["paper_explicit_fields"]["batch_size"] == 10
    assert "train.batch_size=10" in TRAIN


def test_t02_epochs_are_100():
    assert PROTOCOL["paper_explicit_fields"]["epochs_per_stage"] == 100
    assert "train.n_epochs=100" in TRAIN


def test_t03_optimizer_is_adamw():
    assert PROTOCOL["paper_explicit_fields"]["optimizer"] == "torch.optim.AdamW"
    config = (ROOT / "libero/configs/train/optimizer/adam_w.yaml").read_text(encoding="utf-8")
    assert "torch.optim.AdamW" in config


def test_t04_initial_lr_is_1e_4():
    assert PROTOCOL["paper_explicit_fields"]["initial_lr"] == 1e-4
    assert "train.optimizer.kwargs.lr=1e-4" in TRAIN


def test_t05_scheduler_advances_after_optimizer_step():
    assert PROTOCOL["implementation_choices"]["scheduler_step_unit"] == "after_each_optimizer_step"
    base = (ROOT / "libero/lifelong/algos/base.py").read_text(encoding="utf-8")
    optimizer_position = base.index("self.optimizer.step()", base.index("def optimizer_step_and_maybe_schedule"))
    scheduler_position = base.index("self.scheduler.step()", optimizer_position)
    assert scheduler_position > optimizer_position
    assert "+repro_scheduler_mode=batch_step" in TRAIN


def test_t06_warmup_is_500():
    assert PROTOCOL["implementation_choices"]["warmup_steps"] == 500
    scheduler = (ROOT / "libero/configs/train/scheduler/cosine_annealing.yaml").read_text(encoding="utf-8")
    assert "warmup_steps: 500" in scheduler


def test_t07_ifa_lambda_is_point_1():
    assert PROTOCOL["paper_explicit_fields"]["ifa_lambda"] == 0.1
    assert "+ifa_weight=0.1" in TRAIN


def test_t08_ifa_gamma_is_exactly_point_30():
    assert PROTOCOL["paper_explicit_fields"]["ifa_alpha_gamma"] == 0.30
    assert "+gamma=0.30" in TRAIN
    assert "gamma=0.31" not in TRAIN


def test_t09_seeds_are_42_43_44():
    assert PROTOCOL["paper_explicit_fields"]["seeds"] == [42, 43, 44]
    assert "42|43|44" in TRAIN


def test_t10_evaluation_trials_are_20():
    assert PROTOCOL["paper_explicit_fields"]["n_eval"] == 20
    assert "eval.n_eval=20" in EVAL


def test_t11_ep99_is_saved():
    assert PROTOCOL["checkpoint_save_epochs"] == [70, 75, 80, 85, 90, 95, 99]
    assert should_save_mlr_checkpoint(99, 100, (), True)
    assert not should_save_mlr_checkpoint(98, 100, (), True)
    assert "+save_final_checkpoint=true" in TRAIN


def test_t12_primary_evaluation_uses_only_ep99():
    primary = EVAL[EVAL.index('if [[ "$VIEW" == primary ]]'):EVAL.index("else", EVAL.index('if [[ "$VIEW" == primary ]]'))]
    assert "eval_epochs=[99]" in EVAL
    assert "primary_ep99.log" in primary
    assert "70,75,80,85,90,95" not in primary


def test_t13_secondary_cannot_replace_primary():
    assert 'LOG_FILE="$RUN_DIR/primary_ep99.log"' in EVAL
    assert 'LOG_FILE="$RUN_DIR/secondary_selected.log"' in EVAL
    assert 'SENTINEL="$RUN_DIR/PRIMARY_EVAL_COMPLETE"' in EVAL
    assert 'SENTINEL="$RUN_DIR/SECONDARY_EVAL_COMPLETE"' in EVAL
    assert "fixed_epoch_matrix" in SUMMARY
    assert "load_matrix" in SUMMARY


def test_t14_historical_outputs_are_not_overwritten():
    combined = "\n".join((TRAIN, EVAL, RUNNER, SUMMARY))
    assert "runs/final_paper_text_faithful" in combined
    assert "reports/final_paper_text_faithful" in combined
    for historical in (
        "reports/baseline_reconciliation",
        "reports/paper_reproduction_v2",
        "reports/ifa_optimization_interaction",
        "reports/paper_exact_recipe_source_archaeology",
    ):
        assert historical not in combined


def test_t15_methods_share_features_and_pretraining():
    paths = PROTOCOL["paths"]
    assert sorted(paths["feature_cache_seed_mapping"]) == ["42", "43", "44"]
    assert paths["pretrained_checkpoint"].endswith("checkpoints/object_multitask_model_ep99.pth")
    assert TRAIN.count('FEATURE_ROOT="$PROJECT_ROOT/features/object_seed${SEED}"') == 1
    assert 'pretrain="$CHECKPOINT"' in TRAIN
    assert "case \"$METHOD\"" in TRAIN and "IFA_ENABLED=false" in TRAIN and "IFA_ENABLED=true" in TRAIN
