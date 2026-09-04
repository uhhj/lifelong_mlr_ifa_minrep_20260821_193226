#!/usr/bin/env python3
"""Extract public recipe history and reconcile it with paper text and tested runs."""

import argparse
import csv
import json
import subprocess
import urllib.request
from pathlib import Path

from libero.lifelong.analysis.paper_recipe_archaeology import (
    UNKNOWN,
    paper_history_status,
    rank_candidates,
    recipe_from_files,
    relation_to_paper,
    textual_match,
)


TARGETS = (
    "README.md",
    "lifelong_adaptation_object.sbatch",
    "save_feature_object.sbatch",
    "test.sbatch",
    "libero/configs/config_adaption.yaml",
    "libero/configs/test.yaml",
    "libero/configs/eval/default.yaml",
    "libero/configs/train/adaption.yaml",
    "libero/configs/train/default.yaml",
    "libero/configs/train/optimizer/adam_w.yaml",
    "libero/configs/train/scheduler/cosine_annealing.yaml",
    "libero/lifelong/utils.py",
    "libero/lifelong/algos/base.py",
    "libero/lifelong/algos/mlr.py",
    "libero/lifelong/save_feature.py",
    "libero/lifelong/test.py",
    "libero/lifelong/metric.py",
    "libero/lifelong/models/base_policy.py",
    "libero/lifelong/models/brief.py",
    "libero/lifelong/models/policy_head.py",
    "libero/lifelong/models/complete_policy.py",
)

PAPER_CLAIMS = (
    ("batch", "10", "main Sec. 4, implementation paragraph", "high"),
    ("n_epochs", "100", "main Sec. 4, implementation paragraph", "high"),
    ("optimizer", "AdamW", "main Sec. 4, implementation paragraph", "high"),
    ("lr", "0.0001", "main Sec. 4, implementation paragraph", "high"),
    ("scheduler_type", "linear", "main Sec. 4, implementation paragraph", "high"),
    ("ifa_weight", "0.1", "main Sec. 3, total objective", "high"),
    ("alpha", "0.3", "main Sec. 4 and supplement Table 7, OBJECT", "high"),
    ("seeds", "3", "main Sec. 4, evaluation protocol", "high"),
    ("eval_trials", "20", "main Sec. 4, evaluation protocol", "high"),
    ("buffer", "five-demonstration-equivalent", "main Sec. 4, buffer paragraph", "high"),
)


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=check,
    ).stdout


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows and not fieldnames:
        raise ValueError(f"no rows for {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def files_at(repo, commit):
    result, missing = {}, []
    for path in TARGETS:
        exists = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}:{path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if exists:
            result[path] = git(repo, "show", f"{commit}:{path}")
        else:
            missing.append(path)
    return result, missing


def distinct_changes(rows, fields):
    output, previous = [], None
    for row in rows:
        signature = tuple(row.get(field) for field in fields)
        if signature != previous:
            output.append({key: row[key] for key in ("commit", "date", *fields)})
        previous = signature
    return output


def metadata(out):
    path = out / "upstream/upstream_repo_metadata.json"
    try:
        request = urllib.request.Request(
            "https://api.github.com/repos/yfqi/lifelong_mlr_ifa",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "mlr-ifa-archaeology"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = json.load(response)
        payload = {key: raw.get(key) for key in (
            "full_name", "default_branch", "created_at", "updated_at", "pushed_at",
            "license", "forks_count", "open_issues_count", "visibility",
        )}
        payload["status"] = "OK"
    except Exception as exc:  # metadata is optional evidence
        payload = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_metric_sources(root):
    baseline = json.loads((root / "reports/baseline_reconciliation/object_3seed_metrics.json").read_text(encoding="utf-8"))["three_seed_summary"]
    unit_mlr = json.loads((root / "reports/paper_reproduction_v2/mlr_formal/three_seed_summary.json").read_text(encoding="utf-8"))["summary"]
    unit_ifa030 = json.loads((root / "reports/paper_reproduction_v2/ifa_formal/three_seed_summary.json").read_text(encoding="utf-8"))["summary"]
    unit_ifa031 = json.loads((root / "reports/ifa_optimization_interaction/bridge/three_seed_summary.json").read_text(encoding="utf-8"))["summary"]

    def values(summary):
        return {name.upper(): float(summary[name]["mean"]) for name in ("fwt", "nbt", "auc")}

    return {
        "legacy_mlr": values(baseline["mlr"]),
        "legacy_ifa031": values(baseline["mlr_ifa"]),
        "unit_mlr": values(unit_mlr),
        "unit_ifa030": values(unit_ifa030),
        "unit_ifa031": values(unit_ifa031),
    }


def tested_regimes(root, current, candidates, out):
    rows = [
        {"regime": "legacy_mlr", "batch": 10, "scheduler_semantics": "EPOCH_STEP_WITH_BATCH_TOTAL", "warmup": 500, "gamma": "OFF", "feature_aug": "seed-specific saved cache", "eval": "20 trials; ep70:5:95; current-task best"},
        {"regime": "legacy_ifa031", "batch": 10, "scheduler_semantics": "EPOCH_STEP_WITH_BATCH_TOTAL", "warmup": 500, "gamma": 0.31, "feature_aug": "seed-specific saved cache", "eval": "20 trials; ep70:5:95; current-task best"},
        {"regime": "unit_mlr", "batch": 10, "scheduler_semantics": "BATCH_STEP_LINEAR", "warmup": 500, "gamma": "OFF", "feature_aug": "seed-specific saved cache", "eval": "20 trials; ep70:5:95; current-task best"},
        {"regime": "unit_ifa030", "batch": 10, "scheduler_semantics": "BATCH_STEP_LINEAR", "warmup": 500, "gamma": 0.30, "feature_aug": "seed-specific saved cache", "eval": "20 trials; ep70:5:95; current-task best"},
        {"regime": "unit_ifa031", "batch": 10, "scheduler_semantics": "BATCH_STEP_LINEAR", "warmup": 500, "gamma": 0.31, "feature_aug": "seed-specific saved cache", "eval": "20 trials; ep70:5:95; current-task best"},
        {"regime": "paper_text", "batch": 10, "scheduler_semantics": "LINEAR_UNIT_UNSPECIFIED", "warmup": UNKNOWN, "gamma": 0.30, "feature_aug": UNKNOWN, "eval": "3 seeds; 20 trials/seed; selection unspecified"},
        {"regime": "current_release", "batch": current["effective_batch"], "scheduler_semantics": current["scheduler_semantics"], "warmup": current["warmup_steps"], "gamma": current["gamma"], "feature_aug": current["feature_augmentation"], "eval": f"config n_eval={current['eval_trial_count_config']}; epochs={current['evaluation_checkpoint_epochs']}; aggregation unspecified"},
    ]
    for index, candidate in enumerate(candidates, 1):
        rows.append({
            "regime": f"historical_candidate_{index}",
            "batch": candidate["effective_batch"],
            "scheduler_semantics": candidate["scheduler_semantics"],
            "warmup": candidate["warmup_steps"],
            "gamma": candidate["gamma"],
            "feature_aug": candidate["feature_augmentation"],
            "eval": f"n_eval={candidate['eval_trial_count_config']}; epochs={candidate['evaluation_checkpoint_epochs']}",
        })
    write_csv(out / "comparison/tested_regime_recipe_matrix.csv", rows)

    performance = load_metric_sources(root)
    paper = {"MLR": {"FWT": 83.3, "NBT": 12.3, "AUC": 77.6}, "IFA": {"FWT": 84.6, "NBT": 11.4, "AUC": 79.4}}
    perf_rows = []
    for name in ("legacy", "unit030", "unit031"):
        mlr_key = "legacy_mlr" if name == "legacy" else "unit_mlr"
        ifa_key = {"legacy": "legacy_ifa031", "unit030": "unit_ifa030", "unit031": "unit_ifa031"}[name]
        mlr, ifa = performance[mlr_key], performance[ifa_key]
        perf_rows.append({
            "regime": name,
            "MLR_FWT": mlr["FWT"], "MLR_NBT": mlr["NBT"], "MLR_AUC": mlr["AUC"],
            "IFA_FWT": ifa["FWT"], "IFA_NBT": ifa["NBT"], "IFA_AUC": ifa["AUC"],
            "delta_FWT": ifa["FWT"] - mlr["FWT"],
            "delta_NBT": ifa["NBT"] - mlr["NBT"],
            "delta_AUC": ifa["AUC"] - mlr["AUC"],
            "distance_to_paper_mean": round(sum(abs(mlr[m] - paper["MLR"][m]) + abs(ifa[m] - paper["IFA"][m]) for m in ("FWT", "NBT", "AUC")), 3),
            "provenance_note": "performance consistency only; not recipe provenance",
        })
    for candidate in candidates:
        perf_rows.append({
            "regime": f"candidate_{candidate['commit'][:7]}",
            "MLR_FWT": "NOT_TESTED", "MLR_NBT": "NOT_TESTED", "MLR_AUC": "NOT_TESTED",
            "IFA_FWT": "NOT_TESTED", "IFA_NBT": "NOT_TESTED", "IFA_AUC": "NOT_TESTED",
            "delta_FWT": "NOT_TESTED", "delta_NBT": "NOT_TESTED", "delta_AUC": "NOT_TESTED",
            "distance_to_paper_mean": "NOT_TESTED",
            "provenance_note": "not inferred from performance",
        })
    write_csv(out / "comparison/recipe_performance_consistency.csv", perf_rows)


def generate(repo, root, out):
    inventory = read_csv(out / "upstream/commit_inventory.csv")
    rows = []
    for item in inventory:
        files, missing = files_at(repo, item["commit"])
        recipe = recipe_from_files(files)
        changed = git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", item["commit"]).strip().replace("\n", ";")
        rows.append({
            "commit": item["commit"],
            "date": item["commit_date"],
            "author_date": item["author_date"],
            "subject": item["subject"],
            "relation_to_arxiv": relation_to_paper(item["commit_date"]),
            "changed_files": changed,
            "missing_target_count": len(missing),
            "missing_targets": ";".join(missing),
            **recipe,
        })
    write_csv(out / "upstream/commit_recipe_history.csv", rows)

    batch_rows = distinct_changes(rows, ("batch_config", "batch_cli", "effective_batch"))
    gamma_rows = []
    previous_gamma = UNKNOWN
    for row in rows:
        if row["gamma"] == previous_gamma or row["gamma"] == UNKNOWN:
            continue
        gamma_rows.append({
            "commit": row["commit"],
            "date": row["date"],
            "file": "lifelong_adaptation_object.sbatch",
            "symbol": row["gamma_symbol"],
            "value": row["gamma"],
            "line_context": f"+{row['gamma_symbol']}={row['gamma']}",
            "introduced_by_commit": row["commit"],
            "ifa_weight": row["ifa_weight"],
            "paper_symbol": "alpha",
            "semantic_mapping": "same multiplier on angular reference-distance margin",
        })
        previous_gamma = row["gamma"]
    scheduler_rows = []
    previous_scheduler = (UNKNOWN, UNKNOWN, "NO_SCHEDULER")
    for row in rows:
        signature = (row["warmup_steps"], row["total_steps_definition"], row["scheduler_semantics"])
        if signature == previous_scheduler or row["scheduler_semantics"] == "NO_SCHEDULER":
            continue
        scheduler_rows.append({
            "commit": row["commit"],
            "date": row["date"],
            "warmup_steps": row["warmup_steps"],
            "total_steps_definition": row["total_steps_definition"],
            "step_call_file": "libero/lifelong/algos/mlr.py",
            "step_call_scope": "epoch loop after batch loop",
            "semantic_class": row["scheduler_semantics"],
            "notes": "first public code snapshot; unchanged through HEAD",
        })
        previous_scheduler = signature
    aug_rows = distinct_changes(rows, ("feature_augmentation", "feature_stochastic", "feature_seed_controlled", "feature_mode"))
    eval_rows = distinct_changes(rows, ("eval_trial_count_config", "eval_num_procs", "evaluation_checkpoint_epochs", "evaluation_action_semantics", "evaluation_policy_reset"))
    write_csv(out / "upstream/batch_history.csv", batch_rows)
    write_csv(out / "upstream/gamma_history.csv", gamma_rows)
    write_csv(out / "upstream/scheduler_history.csv", scheduler_rows)
    write_csv(out / "upstream/feature_augmentation_history.csv", aug_rows)
    write_csv(out / "upstream/evaluation_history.csv", eval_rows)

    claim_rows = [dict(zip(("claim", "paper_value", "paper_source_location", "confidence"), row)) for row in PAPER_CLAIMS]
    write_csv(out / "paper/paper_claims.csv", claim_rows)
    claims_for_score = {claim: value for claim, value, _, _ in PAPER_CLAIMS if claim not in ("seeds", "buffer")}
    match_rows = []
    for row in rows:
        score = textual_match(row, claims_for_score)
        flags = []
        if row["source_code_available"] in (True, "True"):
            flags.append(row["scheduler_semantics"])
            flags.append(row["feature_augmentation"])
            if str(row["eval_trial_count_config"]) != "20.0":
                flags.append("RELEASE_EVAL_TRIAL_MISMATCH")
        match_rows.append({
            "commit": row["commit"], "date": row["date"], **score,
            "implementation_flags": ";".join(flags) if flags else "NO_PUBLIC_CODE_AT_COMMIT",
        })
    write_csv(out / "upstream/commit_recipe_match.csv", match_rows)

    merged = [{**recipe, **next(match for match in match_rows if match["commit"] == recipe["commit"])} for recipe in rows]
    ranked = rank_candidates(merged)
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
        row["candidate_class"] = "WEAK_CANDIDATE" if row["source_code_available"] in (True, "True") else "NO_PUBLIC_EXACT_RECIPE_CANDIDATE"
        row["limitation"] = "code first published after arXiv; provenance to Table 1 is absent" if row["source_code_available"] in (True, "True") else "README-only commit"
    candidate_fields = (
        "rank", "candidate_class", "commit", "date", "subject", "effective_batch",
        "scheduler_semantics", "gamma", "feature_augmentation", "eval_trial_count_config",
        "evaluation_checkpoint_epochs", "textual_fields_known", "textual_matches",
        "textual_mismatches", "unknowns", "match_fraction", "implementation_flags", "limitation",
    )
    write_csv(out / "comparison/candidate_exact_recipe_commits.csv", ranked, candidate_fields)

    current = rows[-1]
    current_fields = [key for key in current if key not in ("missing_targets", "changed_files")]
    write_csv(out / "upstream/current_release_recipe.csv", [current], current_fields)
    tested_regimes(root, current, ranked, out)
    metadata_payload = metadata(out)

    patch_dir = out / "raw/commit_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    for commit in ("3e5b22f3d1b23a2bf30cbe375c600110dbf302a9", "70e6a1f6b0bd877cf63e3c752d35a0afa970b11e"):
        patch = git(repo, "show", "--format=fuller", commit, "--", *TARGETS)
        (patch_dir / f"{commit}.patch").write_text(patch, encoding="utf-8")

    oldest_author = min(row["author_date"] for row in inventory)
    oldest_commit = min(row["commit_date"] for row in inventory)
    newest_commit = max(row["commit_date"] for row in inventory)
    history_flag = paper_history_status(oldest_author, oldest_commit)
    first_code = next(row for row in rows if row["source_code_available"] in (True, "True"))
    flags = [
        history_flag,
        "PUBLIC_HISTORY_INCOMPLETE_FOR_EXACT_RECIPE",
        "PAPER_BATCH_RELEASE_MISMATCH_CONFIRMED",
        "PAPER_GAMMA_RELEASE_MISMATCH_CONFIRMED",
        "SCHEDULER_UNIT_MISMATCH_PRESENT_AT_HEAD",
        "FEATURE_AUGMENTATION_UNDERSPECIFIED",
        "CHECKPOINT_SELECTION_NOT_PUBLICLY_SPECIFIED",
        "EVALUATION_PROTOCOL_PARTIALLY_SPECIFIED",
        "NO_HISTORICAL_COMMIT_MATCHES_ALL_PAPER_TEXT",
        "NO_PUBLIC_PAPER_TIME_SNAPSHOT",
    ]
    verdict = "PAPER_EXACT_RECIPE_NOT_PUBLICLY_RECOVERABLE"

    timeline_lines = [
        "# Source timeline", "", "| Date | Commit | Relation | Batch | Scheduler | Gamma | Feature aug | Eval |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    for row in rows:
        timeline_lines.append(
            f"| {row['date']} | `{row['commit'][:7]}` | {row['relation_to_arxiv']} | {row['effective_batch']} | {row['scheduler_semantics']} | {row['gamma']} | {row['feature_augmentation']} | n_eval={row['eval_trial_count_config']}; epochs={row['evaluation_checkpoint_epochs']} |"
        )
    (out / "SOURCE_TIMELINE.md").write_text("\n".join(timeline_lines) + "\n", encoding="utf-8")

    author_query = """# Author query draft

1. Did the Table 1 LIBERO-OBJECT runs use batch size 10 as written in the paper, or batch size 18 from the released adaptation config?
2. Was the 500-step linear scheduler stepped per optimizer update or once per epoch?
3. Was the OBJECT angular factor 0.30 or 0.31?
4. Were replay latent features saved with stochastic augmentation enabled?
5. Which public or internal commit corresponds to Table 1, and how were checkpoints selected for its 20-trial evaluation?

This draft has not been sent.
"""
    (out / "AUTHOR_QUERY_DRAFT.md").write_text(author_query, encoding="utf-8")

    best = ranked[0]
    report = f"""# Paper Exact-Recipe Source Archaeology

## Verdict

**{verdict}**

## Flags

{', '.join(flags)}

## Branch / SHA

- Starting SHA: `419ecfb366b77207e44efb73667a1e5cfc7f3040`
- Branch: `research/paper-exact-recipe-source-archaeology`
- Execution scope: static Git/paper inspection only; no training, optimizer step, feature extraction, rollout, or GPU execution.

## Public repository history

The public repository has {len(inventory)} reachable commits. The oldest author date is {oldest_author}, the oldest commit date is {oldest_commit}, and the newest commit date is {newest_commit}. Although README-only history exists before arXiv, the first public source-code snapshot is `{first_code['commit']}` dated {first_code['date']}, after the arXiv submissions. GitHub metadata status: {metadata_payload.get('status')}.

## Paper timeline

- arXiv v1: 2026-03-11
- arXiv v2: 2026-03-12
- CVPR publication: June 2026

## Paper textual recipe

The paper explicitly states batch 10, 100 epochs, AdamW with initial LR 1e-4 and a linear scheduler, IFA weight 0.1, OBJECT alpha 0.30, three seeds, 20 trials per seed, and a five-demonstration-equivalent latent buffer. It does not state the scheduler step unit, feature-augmentation setting, or checkpoint-selection rule.

## Current release recipe

Static parsing gives effective batch {current['effective_batch']}, 100 epochs, AdamW LR {current['lr']} with weight decay {current['weight_decay']}, warmup {current['warmup_steps']}, scheduler semantics **{current['scheduler_semantics']}**, release gamma {current['gamma']}, feature extraction **{current['feature_augmentation']}**, evaluation config n_eval={current['eval_trial_count_config']}, checkpoints {current['evaluation_checkpoint_epochs']}, and action semantics {current['evaluation_action_semantics']}.

## Historical recipe timeline

No source code exists in the public history until 2026-06-03. The June 15 source change only alters evaluation logging/comments and the default benchmark/checkpoint in `test.sbatch`; it does not establish a different Table 1 recipe.

## Batch history

The only public code snapshot uses adaptation-config batch 18 and provides no OBJECT launcher override. This conflicts with paper batch 10.

## Scheduler history

Every public code snapshot computes total steps as epochs × loader length but calls `scheduler.step()` once after each epoch: **EPOCH_STEP_WITH_BATCH_TOTAL**. No public batch-step snapshot exists, so the mismatch cannot be attributed to a post-paper cleanup change from public history.

## Gamma history

Every public OBJECT launcher snapshot uses config symbol `gamma=0.31`; the loss multiplies the same angular reference distance described as alpha in the paper. Thus gamma and alpha are semantically corresponding factors, and public code conflicts with paper alpha 0.30.

## Feature extraction history

Every public feature launcher sets `+aug=True`; `train.use_augmentation=true`, and preprocessing invokes stochastic image transforms. `control_seed(cfg.seed)` seeds the process, so extraction is stochastic but seed-controlled. The paper does not specify feature augmentation.

## Evaluation history

The release evaluates epochs 70,75,80,85,90,95 but exposes no official aggregation or checkpoint-selection procedure. Its default `n_eval=2`, despite paper 20 trials. Initial-state indexing is fixed, environment seeding is applied, and `CompletePolicy.get_action` samples from the GMM, so actions are stochastic. The released test loop resets the global seed once per seed before sequential checkpoint evaluation, not before every checkpoint.

## Candidate paper-time commits

The highest textual-match public candidate is `{best['commit']}` ({best['date']}, fraction {best['match_fraction']}), but it is only a **WEAK_CANDIDATE**: its code was published after arXiv and retains batch/gamma/evaluation conflicts. No public paper-time source candidate exists.

## Comparison with tested regimes

The closest scientifically coherent tested regime is unit-consistent MLR / IFA gamma0.31. Performance similarity is recorded only as a consistency check and is not used as provenance evidence. No existing or historical-public recipe is an exact evidence-backed match to the published recipe.

## What is directly supported

Paper textual values, all nine public commits, the first-code publication date, current release batch/scheduler/gamma/augmentation/evaluation behavior, and the absence of a public checkpoint-selection rule are directly supported by retained evidence.

## What is inferred

The June release is a cleanup-era snapshot rather than a demonstrated Table 1 snapshot. Unit-consistent gamma0.31 is recommended as the local research baseline because its scheduler semantics are coherent and its IFA direction matches the paper; this is a baseline choice, not a claim about the authors' private recipe.

## What cannot be recovered publicly

The Table 1 internal commit, scheduler step unit, feature augmentation choice, exact 20-trial implementation, and checkpoint-selection/aggregation rule cannot be recovered from public paper-time source because no such source snapshot is public.

## Final exact-recipe assessment

Method-level reproduction is supported; exact published recipe is not publicly recoverable from current release/history.

## Recommended next action

Stop archaeology and send `AUTHOR_QUERY_DRAFT.md` only after user approval. Until new provenance appears, reuse unit-consistent MLR and unit-consistent IFA gamma0.31; do not launch another parameter sweep or infer recipe from performance.
"""
    (out / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return {"verdict": verdict, "flags": flags, "commits": len(inventory), "best": best["commit"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.repo.resolve(), args.root.resolve(), args.out.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
