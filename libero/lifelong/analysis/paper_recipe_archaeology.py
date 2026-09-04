"""Static helpers for paper-recipe source archaeology.

This module deliberately contains no model, environment, or optimizer execution.
"""

import math
import re
from datetime import datetime


UNKNOWN = "UNKNOWN"


def _scalar(text, key):
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*([^#\n]+)", text or "")
    return match.group(1).strip().strip("\"'") if match else UNKNOWN


def parse_yaml_batch(text):
    value = _scalar(text, "batch_size")
    return int(value) if value != UNKNOWN and value.isdigit() else UNKNOWN


def parse_cli_value(text, key):
    patterns = (
        rf"(?:^|\s){re.escape(key)}=([^\s\\]+)",
        rf"(?:^|\s)\+{re.escape(key)}=([^\s\\]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", re.MULTILINE)
        if match:
            return match.group(1).strip("\"'")
    return UNKNOWN


def effective_batch(config_text, launcher_text):
    override = parse_cli_value(launcher_text, "train.batch_size")
    if override != UNKNOWN:
        return int(override)
    return parse_yaml_batch(config_text)


def classify_scheduler(source_text):
    text = source_text or ""
    if "scheduler.step()" not in text:
        return "NO_SCHEDULER"
    batch_total = bool(
        re.search(r"n_epochs\s*\*\s*len\s*\(\s*train_(?:data)?loader\s*\)", text)
    )
    epoch_total = bool(re.search(r"total_steps\s*=\s*(?:self\.)?cfg\.train\.n_epochs(?!\s*\*)", text))

    lines = text.splitlines()
    step_scope = UNKNOWN
    for index, line in enumerate(lines):
        if "scheduler.step()" not in line:
            continue
        step_indent = len(line) - len(line.lstrip())
        epoch_indent = None
        batch_indent = None
        for previous in reversed(lines[:index]):
            stripped = previous.strip()
            indent = len(previous) - len(previous.lstrip())
            if batch_indent is None and re.match(r"for .*\bdata\b.* in .*loader", stripped):
                batch_indent = indent
            if re.match(r"for epoch\b", stripped):
                epoch_indent = indent
                break
        effective_indent = step_indent
        for previous in reversed(lines[:index]):
            stripped = previous.strip()
            indent = len(previous) - len(previous.lstrip())
            if indent < step_indent and stripped.startswith("if ") and "scheduler" in stripped:
                effective_indent = indent
                break
            if indent < step_indent and stripped:
                break
        if batch_indent is not None and effective_indent > batch_indent:
            step_scope = "batch"
        elif epoch_indent is not None and effective_indent == epoch_indent + 4:
            step_scope = "epoch"
        else:
            context = "\n".join(lines[max(0, index - 8):index + 2])
            step_scope = "batch" if "optimizer.step()" in context else "epoch"
        break

    if step_scope == "batch" and batch_total:
        return "BATCH_STEP_LINEAR"
    if step_scope == "epoch" and batch_total:
        return "EPOCH_STEP_WITH_BATCH_TOTAL"
    if step_scope == "epoch" and epoch_total:
        return "EPOCH_STEP_EPOCH_TOTAL"
    return "OTHER"


def parse_gamma(text):
    for key in ("gamma", "alpha"):
        value = parse_cli_value(text, key)
        if value != UNKNOWN:
            try:
                return float(value), key
            except ValueError:
                pass
    match = re.search(r"(?m)^\s*(?:gamma|alpha)\s*[:=]\s*([0-9.]+)", text or "")
    return (float(match.group(1)), "source") if match else (UNKNOWN, UNKNOWN)


def parse_augmentation(launcher_text, config_text=""):
    value = parse_cli_value(launcher_text, "aug")
    if value == UNKNOWN:
        value = _scalar(config_text, "use_augmentation")
    lowered = str(value).lower()
    if lowered in ("true", "1", "yes", "on"):
        return "FEATURE_AUG_ON"
    if lowered in ("false", "0", "no", "off"):
        return "FEATURE_AUG_OFF"
    return "AMBIGUOUS"


def relation_to_paper(date_text):
    value = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
    day = value.date().isoformat()
    if day < "2026-03-11":
        return "BEFORE_ARXIV"
    if day <= "2026-03-12":
        return "ON_ARXIV_WINDOW"
    if day < "2026-06-05":
        return "AFTER_ARXIV"
    return "AFTER_CVPR_PUBLICATION"


def paper_history_status(oldest_author_date, oldest_commit_date):
    cutoff = "2026-03-12"
    author_day = oldest_author_date[:10]
    commit_day = oldest_commit_date[:10]
    if author_day <= cutoff and commit_day <= cutoff:
        return "PUBLIC_HISTORY_PRE_PAPER_EXISTS"
    if author_day > cutoff and commit_day > cutoff:
        return "PUBLIC_HISTORY_STARTS_AFTER_PAPER"
    return "PUBLIC_HISTORY_REWRITTEN_OR_INCOMPLETE"


def textual_match(recipe, claims):
    aliases = {
        "alpha": "gamma",
        "batch": "effective_batch",
        "scheduler_type": "scheduler_name",
        "eval_trials": "eval_trial_count_config",
    }
    known = matches = mismatches = 0
    unknowns = []
    for claim, expected in claims.items():
        field = aliases.get(claim, claim)
        actual = recipe.get(field, UNKNOWN)
        if actual in (UNKNOWN, "", None, "MISSING_AT_COMMIT"):
            unknowns.append(claim)
            continue
        known += 1
        try:
            equal = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            equal = str(actual).lower() == str(expected).lower()
        if equal:
            matches += 1
        else:
            mismatches += 1
    return {
        "textual_fields_known": known,
        "textual_matches": matches,
        "textual_mismatches": mismatches,
        "unknowns": ";".join(unknowns),
        "match_fraction": round(matches / known, 6) if known else 0.0,
    }


def rank_candidates(rows, paper_date="2026-03-11"):
    target = datetime.fromisoformat(paper_date).astimezone() if "T" in paper_date else datetime.fromisoformat(paper_date)

    def key(row):
        moment = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
        if moment.tzinfo is not None and target.tzinfo is None:
            moment = moment.replace(tzinfo=None)
        return (
            -float(row.get("match_fraction", 0)),
            int(row.get("textual_mismatches", 0)),
            abs((moment - target).total_seconds()),
            row["commit"],
        )

    return sorted(rows, key=key)[:3]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return UNKNOWN


def recipe_from_files(files):
    """Extract one static recipe row from a mapping of path -> file text."""
    adapt = files.get("libero/configs/train/adaption.yaml", "")
    launcher = files.get("lifelong_adaptation_object.sbatch", "")
    feature_launcher = files.get("save_feature_object.sbatch", "")
    optimizer = files.get("libero/configs/train/optimizer/adam_w.yaml", "")
    base = files.get("libero/lifelong/algos/base.py", "")
    mlr = files.get("libero/lifelong/algos/mlr.py", "")
    utils = files.get("libero/lifelong/utils.py", "")
    evaluation = files.get("libero/lifelong/test.py", "")
    eval_config = files.get("libero/configs/eval/default.yaml", "")
    complete_policy = files.get("libero/lifelong/models/complete_policy.py", "")

    gamma, gamma_symbol = parse_gamma(launcher)
    save_epochs = sorted({
        int(value)
        for group in re.findall(r"epoch\s*in\s*\[([^]]+)\]", mlr)
        for value in re.findall(r"\d+", group)
    })
    if not save_epochs and re.search(r"epoch\s*%\s*5\s*==\s*0", mlr) and re.search(r"epoch\s*>\s*69", mlr):
        save_epochs = list(range(70, 100, 5))
    eval_range = re.search(r"range\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", evaluation)
    eval_epochs = []
    if eval_range:
        eval_epochs = list(range(*map(int, eval_range.groups())))
    if "dist.sample()" in complete_policy or "action = dist.sample()" in complete_policy:
        action = "STOCHASTIC_GMM_SAMPLE"
    elif ".mode(" in complete_policy:
        action = "HIGHEST_WEIGHT_COMPONENT_MEAN"
    else:
        action = UNKNOWN
    total = "n_epochs*len(train_loader)" if re.search(r"n_epochs\s*\*\s*len\s*\(\s*train_(?:data)?loader", base + mlr) else UNKNOWN
    scheduler_sources = "\n".join((base, mlr, utils))
    return {
        "source_code_available": bool(base and mlr),
        "batch_config": parse_yaml_batch(adapt),
        "batch_cli": parse_cli_value(launcher, "train.batch_size"),
        "effective_batch": effective_batch(adapt, launcher),
        "n_epochs": safe_float(_scalar(adapt, "n_epochs")),
        "optimizer": "AdamW" if "AdamW" in optimizer or "optim.AdamW" in optimizer else UNKNOWN,
        "lr": safe_float(_scalar(optimizer, "lr")),
        "betas": _scalar(optimizer, "betas"),
        "eps": safe_float(_scalar(optimizer, "eps")),
        "weight_decay": safe_float(_scalar(optimizer, "weight_decay")),
        "scheduler_name": "linear" if "make_linear_schedule_with_warmup" in scheduler_sources else UNKNOWN,
        "warmup_steps": int(_scalar(files.get("libero/configs/train/scheduler/cosine_annealing.yaml", ""), "warmup_steps")) if _scalar(files.get("libero/configs/train/scheduler/cosine_annealing.yaml", ""), "warmup_steps").isdigit() else UNKNOWN,
        "total_steps_definition": total,
        "scheduler_semantics": classify_scheduler(scheduler_sources),
        "start_task": safe_float(parse_cli_value(launcher, "start_task")),
        "end_task": safe_float(parse_cli_value(launcher, "end_task")),
        "past_data_rate": safe_float(parse_cli_value(launcher, "past_data_rate")),
        "gamma": gamma,
        "gamma_symbol": gamma_symbol,
        "ifa_weight": 0.1 if re.search(r"loss\s*=\s*loss\s*\+\s*0\.1\s*\*\s*loss_hinge", base) else UNKNOWN,
        "pair_selection_ratio": 0.5 if "top_ratio=0.5" in mlr or "top_ratio = 0.5" in mlr else UNKNOWN,
        "feature_augmentation": parse_augmentation(feature_launcher, adapt),
        "feature_stochastic": "YES" if "DataAugGroup" in files.get("libero/lifelong/models/base_policy.py", "") else UNKNOWN,
        "feature_seed_controlled": "YES_GLOBAL_SEED" if "control_seed(cfg.seed)" in files.get("libero/lifelong/save_feature.py", "") else UNKNOWN,
        "feature_mode": "TRAINING_FLAG_FALSE_BUT_PREPROCESS_AUG_ACTIVE" if "+aug=True" in feature_launcher else UNKNOWN,
        "checkpoint_save_epochs": ";".join(map(str, save_epochs)) if save_epochs else UNKNOWN,
        "eval_trial_count_config": safe_float(_scalar(eval_config, "n_eval")),
        "eval_num_procs": safe_float(_scalar(eval_config, "num_procs")),
        "evaluation_checkpoint_epochs": ";".join(map(str, eval_epochs)) if eval_epochs else UNKNOWN,
        "evaluation_action_semantics": action,
        "evaluation_policy_reset": "YES_ALGO_RESET" if "algo.reset()" in files.get("libero/lifelong/metric.py", "") or "algo.policy.reset()" in files.get("libero/lifelong/metric.py", "") else UNKNOWN,
    }
