#!/usr/bin/env python3
"""Install three additive experiment files and one registry import into MLR."""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMPORT = "from libero.lifelong.algos.mlr_replay_constraint import MLRReplayConstraint"
FILES = {
    "replay_constraint.py": "libero/lifelong/algos/replay_constraint.py",
    "mlr_replay_constraint.py": "libero/lifelong/algos/mlr_replay_constraint.py",
    "mlr_replay_constraint.yaml": "libero/configs/lifelong/mlr_replay_constraint.yaml",
}


def plan(repo):
    repo = Path(repo).resolve()
    required = {
        "libero/lifelong/algos/mlr.py": ("class MLR", "build_dataset_for_task", "save_final_checkpoint"),
        "libero/lifelong/algos/base.py": ("def observe", "optimizer_step_and_maybe_schedule", "batch_step"),
        "libero/lifelong/models/base_policy.py": ("def compute_loss", "reduction",),
        "libero/lifelong/models/brief.py": ("class BriefPolicy", "temporal_transformer", "policy_head"),
        "libero/lifelong/models/policy_head.py": ('reduction == "none"',),
        "libero/lifelong/algos/__init__.py": ("get_algo_class",),
        "libero/lifelong/test.py": ("eval_epochs",),
    }
    for rel, tokens in required.items():
        path = repo / rel
        if not path.is_file():
            raise ValueError("Missing full baseline file: " + str(path))
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(path))
        for token in tokens:
            if token not in source:
                raise ValueError("Unsupported baseline API: %s lacks %r" % (path, token))
    changes = []
    for source, destination in FILES.items():
        content = (ROOT / "src" / source).read_text(encoding="utf-8")
        path = repo / destination
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Refusing to overwrite different experiment file: " + str(path))
        if not path.exists():
            changes.append((path, content))
    registry = repo / "libero/lifelong/algos/__init__.py"
    content = registry.read_text(encoding="utf-8-sig")
    if IMPORT not in content.splitlines():
        changes.append((registry, content.rstrip() + "\n" + IMPORT + "\n"))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changes = plan(args.repo)
    for path, content in changes:
        print(("Would write: " if args.dry_run else "Writing: ") + str(path))
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
    print("Compatible baseline; %d changes%s." % (len(changes), " planned" if args.dry_run else " applied"))


if __name__ == "__main__":
    main()
