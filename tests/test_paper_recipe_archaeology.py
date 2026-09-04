from pathlib import Path

from libero.lifelong.analysis.paper_recipe_archaeology import (
    UNKNOWN,
    classify_scheduler,
    effective_batch,
    paper_history_status,
    parse_augmentation,
    parse_gamma,
    parse_yaml_batch,
    rank_candidates,
    recipe_from_files,
    relation_to_paper,
    textual_match,
)


def test_parse_yaml_batch():
    assert parse_yaml_batch("batch_size: 18\n") == 18


def test_cli_batch_override_wins():
    assert effective_batch("batch_size: 18\n", "train.batch_size=10") == 10


def test_classify_batch_step_linear():
    source = "total_steps = cfg.train.n_epochs * len(train_loader)\nfor data in train_loader:\n    optimizer.step()\n    scheduler.step()\n"
    assert classify_scheduler(source) == "BATCH_STEP_LINEAR"


def test_classify_epoch_step_with_batch_total():
    source = "total_steps = cfg.train.n_epochs * len(train_loader)\nfor epoch in range(100):\n    for data in train_loader:\n        optimizer.step()\n    scheduler.step()\n"
    assert classify_scheduler(source) == "EPOCH_STEP_WITH_BATCH_TOTAL"


def test_parse_gamma():
    assert parse_gamma("+gamma=0.31 \\\n")[0] == 0.31


def test_parse_augmentation():
    assert parse_augmentation("+aug=True", "use_augmentation: true") == "FEATURE_AUG_ON"


def test_commit_recipe_row():
    files = {
        "libero/configs/train/adaption.yaml": "n_epochs: 100\nbatch_size: 18\n",
        "lifelong_adaptation_object.sbatch": "+gamma=0.31 +start_task=6 +end_task=10",
    }
    row = recipe_from_files(files)
    assert row["effective_batch"] == 18
    assert row["gamma"] == 0.31


def test_textual_match_score():
    score = textual_match({"effective_batch": 10, "gamma": 0.3}, {"batch": "10", "alpha": "0.3"})
    assert score["textual_matches"] == 2
    assert score["match_fraction"] == 1.0


def test_date_classification():
    assert relation_to_paper("2026-03-11T12:00:00+00:00") == "ON_ARXIV_WINDOW"
    assert relation_to_paper("2026-06-03T12:00:00+00:00") == "AFTER_CVPR_PUBLICATION"


def test_missing_files_graceful():
    row = recipe_from_files({})
    assert row["effective_batch"] == UNKNOWN
    assert row["source_code_available"] is False


def test_candidate_ranking_deterministic():
    rows = [
        {"commit": "b", "date": "2026-03-12T00:00:00+00:00", "match_fraction": 0.5, "textual_mismatches": 1},
        {"commit": "a", "date": "2026-03-10T00:00:00+00:00", "match_fraction": 0.5, "textual_mismatches": 1},
    ]
    assert [row["commit"] for row in rank_candidates(rows)] == ["a", "b"]


def test_runner_contains_no_execution_entrypoints():
    text = (Path(__file__).parents[1] / "scripts/run_paper_exact_recipe_archaeology.sh").read_text(encoding="utf-8")
    forbidden = ("lifelong_adaptation.py", "save_feature.py", "libero/lifelong/test.py", "optimizer.step", "rollout")
    assert all(token not in text for token in forbidden)


def test_one_commit_history():
    row = {"commit": "only", "date": "2026-06-03T00:00:00+00:00", "match_fraction": 0.2, "textual_mismatches": 2}
    assert rank_candidates([row]) == [row]


def test_author_commit_date_discrepancy():
    assert paper_history_status("2026-03-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00") == "PUBLIC_HISTORY_REWRITTEN_OR_INCOMPLETE"


def test_no_scheduler_classification():
    assert classify_scheduler("optimizer.step()") == "NO_SCHEDULER"
