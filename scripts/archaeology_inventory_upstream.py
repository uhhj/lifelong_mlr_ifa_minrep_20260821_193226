#!/usr/bin/env python3
"""Inventory all reachable public upstream commits without executing project code."""

import argparse
import csv
import subprocess
from pathlib import Path


def git(repo, *args, check=True):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )
    return result.stdout


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def inventory(repo, out):
    raw = out / "raw"
    upstream = out / "upstream"
    raw.mkdir(parents=True, exist_ok=True)
    upstream.mkdir(parents=True, exist_ok=True)

    write_text(raw / "git_log_graph.txt", git(repo, "log", "--all", "--graph", "--decorate", "--oneline", "--date-order"))
    write_text(raw / "git_log_full.txt", git(repo, "log", "--all", "--pretty=fuller", "--date=iso-strict"))
    write_text(raw / "git_branches.txt", git(repo, "branch", "-a"))
    write_text(raw / "git_tags.txt", git(repo, "tag", "--list"))
    write_text(raw / "git_fsck.txt", git(repo, "fsck", "--full", "--no-reflogs", "--unreachable", check=False))
    write_text(raw / "git_remote.txt", git(repo, "remote", "-v"))
    searches = []
    for label, args in (
        ("gamma_031", ("log", "-S0.31", "--all", "--", ".")),
        ("gamma_03", ("log", "-S0.3", "--all", "--", ".")),
        ("ifa_symbols", ("log", "-Ggamma|alpha|ifa_weight", "--all", "--", ".")),
        ("scheduler_step", ("log", "-Sscheduler.step()", "--all", "--", "libero/lifelong/algos/base.py", "libero/lifelong/algos/mlr.py")),
        ("scheduler_symbols", ("log", "-Gscheduler\\.step|warmup_steps|total_steps|LambdaLR|linear", "--all", "--", ".")),
        ("augmentation", ("log", "-Gaug=True|aug=False|use_augmentation|preprocess_input", "--all", "--", ".")),
        ("checkpoint_epochs", ("log", "-G70|75|80|85|90|95", "--all", "--", ".")),
    ):
        searches.append(f"===== {label} =====\n{git(repo, *args, check=False)}")
    write_text(raw / "git_recipe_searches.txt", "\n".join(searches))

    fmt = "%H%x09%P%x09%an%x09%ae%x09%aI%x09%cI%x09%s"
    lines = git(repo, "log", "--all", "--reverse", f"--pretty=format:{fmt}").splitlines()
    rows = []
    for line in lines:
        parts = line.split("\t", 6)
        if len(parts) != 7:
            continue
        commit = parts[0]
        reachable = git(repo, "branch", "-a", "--contains", commit).replace("*", "").strip().replace("\n", ";")
        rows.append(dict(zip(
            ("commit", "parents", "author_name", "author_email", "author_date", "commit_date", "subject"),
            parts,
        )))
        rows[-1]["reachable_from"] = reachable
    path = upstream / "commit_inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = inventory(args.repo.resolve(), args.out.resolve())
    print(f"inventoried {len(rows)} public commits")


if __name__ == "__main__":
    main()
