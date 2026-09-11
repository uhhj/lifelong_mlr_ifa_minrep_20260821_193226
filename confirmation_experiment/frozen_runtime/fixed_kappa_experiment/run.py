#!/usr/bin/env python3
"""Fixed-ep99 LIBERO-OBJECT experiment launcher. Uses the existing Python env."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from metrics import read_fixed_matrix, score_matrix
from protocol import audit_references, read, require, sha, source_hashes


def arm_name(kappa):
    import math
    if isinstance(kappa, bool) or not math.isfinite(kappa) or not 0 <= kappa <= 1:
        raise ValueError("kappa must be finite and in [0, 1]")
    return "kappa_" + format(kappa, ".15g").replace(".", "p")


def configuration(args):
    repo = args.repo.resolve()
    args.arm = arm_name(args.kappa)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("tag must contain only letters, numbers, '_' or '-'.")
    phase = "smoke" if args.smoke else "formal"
    run = repo / "runs" / "fixed_kappa" / args.tag / phase / args.arm / ("seed%d" % args.seed)
    buffer = repo / "buffers" / "fixed_kappa" / args.tag / phase / args.arm / ("seed%d" % args.seed)
    feature = (args.feature_root or repo / "features" / ("object_seed%d" % args.seed)).resolve()
    data = (args.data_root or repo / "data").resolve()
    checkpoint = (args.checkpoint or repo / "checkpoints/object_multitask_model_ep99.pth").resolve()
    return repo, run, buffer, feature, data, checkpoint


def commands(args):
    repo, run, buffer, feature, data, checkpoint = configuration(args)
    py = args.python or sys.executable
    if args.action == "train":
        cmd = [py, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
               "libero/lifelong/lifelong_adaptation.py",
               "seed=%d" % args.seed, "benchmark_name=LIBERO_OBJECT", "policy=BriefPolicy",
               "lifelong=mlr_fixed_kappa",
               "folder=" + str(data), "pretrain=" + str(checkpoint),
               "train.n_epochs=%d" % (1 if args.smoke else 100),
               "train.batch_size=10", "train.num_workers=0", "train.optimizer.kwargs.lr=1e-4",
               "train.optimizer.kwargs.betas=[0.9,0.999]", "train.optimizer.kwargs.eps=1e-8",
               "train.optimizer.kwargs.weight_decay=0.1",
               "+start_task=6", "+end_task=%d" % (8 if args.smoke else 10), "+aug=false",
               "+feature_dir=" + str(feature), "+buffer_dir=" + str(buffer),
               "+experiment_root=" + str(run), "+past_data_rate=0.5",
               "+ifa_enabled=false",
               "+ifa_weight=0.1", "+gamma=0.30",
               "+repro_scheduler_mode=" + args.scheduler_mode,
               "+repro_arm=fixed_kappa_" + args.arm,
               "+save_final_checkpoint=true", "training=true"]
        cmd += ["lifelong.projection_enabled=true", "lifelong.kappa=" + str(args.kappa)]
        if args.smoke:
            cmd += ["+max_optimizer_updates=4"]
    else:
        # A single unchanged evaluator call preserves historical task/RNG order.
        cmd = [py, "libero/lifelong/test.py", "seed=%d" % args.seed,
               "benchmark_name=LIBERO_OBJECT", "policy=complete_policy", "lifelong=basic",
               "folder=" + str(data), "pretrain=" + str(checkpoint),
               "bddl_folder=" + str(repo / "libero/libero/bddl_files"),
               "init_states_folder=" + str(repo / "libero/libero/init_files"),
               "eval.n_eval=20", "eval.num_procs=20", "+start_task=6", "+end_task=10",
               "+aug=true", "+epoch_dir=" + str(run / "run_001"),
               "+test_log_file=" + str(run / "primary_ep99.log"),
               "+eval_epochs=[99]", "+experiment_root=" + str(run / "eval_primary_ep99"),
               "training=false"]
    return cmd


def execute(args):
    repo, run, buffer, feature, data, checkpoint = configuration(args)
    cmd = commands(args)
    if args.dry_run:
        print(json.dumps({"command": cmd, "cwd": str(repo), "run_dir": str(run),
                          "buffer_dir": str(buffer), "gpu": args.gpu}, indent=2))
        return
    for path in (repo / "libero/lifelong/lifelong_adaptation.py", checkpoint):
        if not path.is_file():
            raise ValueError("Missing input: " + str(path))
    if not (data / "libero_object").is_dir():
        raise ValueError("Missing LIBERO-OBJECT data.")
    if not (repo / "libero/lifelong/algos/mlr_fixed_kappa.py").is_file():
        raise ValueError("Install this experiment into the baseline first.")
    for name in ('fixed_kappa.py', 'mlr_fixed_kappa.py'):
        require(sha(Path(__file__).parent / 'src' / name) == sha(repo / 'libero/lifelong/algos' / name),
                'Installed experiment source differs: ' + name)
    base_source = (repo / "libero/lifelong/algos/base.py").read_text(encoding="utf-8-sig")
    if args.scheduler_mode not in base_source:
        raise ValueError("Baseline does not support requested scheduler mode.")
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, MUJOCO_EGL_DEVICE_ID="0", MUJOCO_GL="egl",
               PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", WANDB_MODE="disabled",
               HYDRA_FULL_ERROR="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", str(repo / "data/huggingface_cache"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Reuse baseline path configuration; never overwrite its assets/dataset config.
    env.setdefault("LIBERO_CONFIG_PATH", str(repo / "data/libero_config"))
    if not Path(env["LIBERO_CONFIG_PATH"]).is_dir():
        raise ValueError("Set LIBERO_CONFIG_PATH to your existing working baseline configuration.")
    if args.action == "train":
        references = audit_references(repo, args.seed, feature, checkpoint)
        for stage in range(6, 8 if args.smoke else 10):
            if not any((feature / str(stage)).glob("*.pt")):
                raise ValueError("Missing seed-specific feature files for task %d" % stage)
        for destination in (run, buffer):
            if destination.exists() and any(destination.iterdir()):
                raise ValueError("Refusing nonempty experiment destination: " + str(destination))
        run.mkdir(parents=True, exist_ok=True)
        buffer.mkdir(parents=True, exist_ok=True)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unavailable"
        manifest = {"arm": args.arm, "kappa": args.kappa, "seed": args.seed, "smoke": args.smoke,
                    "scheduler_mode": args.scheduler_mode, "evaluation_epoch": 99,
                    "evaluation_trials": 20, "source_revision": revision,
                    "feature_root": str(feature), "checkpoint": str(checkpoint),
                    "data_root": str(data), "command": cmd,
                    "references": references, "source_sha256": source_hashes(repo)}
        (run / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        if args.smoke:
            raise ValueError("Smoke checkpoints are not eligible for formal ep99 evaluation.")
        if not (run / "TRAINING_COMPLETE").is_file():
            raise ValueError("Training has not completed.")
        manifest = read(run / 'experiment.json')
        require(manifest['kappa'] == args.kappa and manifest['seed'] == args.seed, 'Wrong trained run')
        require(read(run / 'protocol_validation.json')['status'] == 'PASS', 'Validate training before evaluation')
        require(manifest['source_sha256'] == source_hashes(repo), 'Source changed since training')
        require(manifest['references']['initial_checkpoint_sha256'] == sha(checkpoint), 'Initial checkpoint changed')
        for stage in range(6, 10):
            if not (run / "run_001" / ("mlrifa_%d_model_ep99.pth" % stage)).is_file():
                raise ValueError("Missing ep99 checkpoint for stage %d" % stage)
        if (run / "primary_ep99.log").exists() or (run / "eval_stdout.log").exists():
            raise ValueError("Evaluation output exists; inspect it instead of appending duplicate trials.")
        (run / 'evaluation_protocol.json').write_text(json.dumps({'command': cmd, 'kappa': args.kappa,
            'checkpoint_sha256': {str(stage): sha(run / 'run_001' / ('mlrifa_%d_model_ep99.pth' % stage))
                                  for stage in range(6, 10)}}, indent=2) + '\n', encoding='utf-8')
    log = run / ("train_stdout.log" if args.action == "train" else "eval_stdout.log")
    started = time.perf_counter()
    print("Running %s: %s" % (args.action, run), flush=True)
    print("Live output: " + str(log), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(cmd, cwd=str(repo), env=env, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    (run / (args.action + "_execution.json")).write_text(
        json.dumps({"returncode": code, "wall_seconds": elapsed}, indent=2) + "\n", encoding="utf-8")
    if code:
        raise RuntimeError("%s failed (%d); see %s" % (args.action, code, log))
    if args.action == "train":
        end, ep = (8, 0) if args.smoke else (10, 99)
        for stage in range(6, end):
            checkpoint_out = run / "run_001" / ("mlrifa_%d_model_ep%d.pth" % (stage, ep))
            if not checkpoint_out.is_file() or checkpoint_out.stat().st_size == 0:
                raise RuntimeError("Training returned without expected checkpoint: " + str(checkpoint_out))
        (run / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    else:
        matrix = read_fixed_matrix(run / "primary_ep99.log")
        protocol = read(run / 'evaluation_protocol.json')
        require(protocol['checkpoint_sha256'] == {str(s): sha(run / 'run_001' / ('mlrifa_%d_model_ep99.pth' % s))
                                                  for s in range(6,10)}, 'Checkpoint changed during evaluation')
        (run / 'evaluation_complete.json').write_text(json.dumps({
            'log_sha256': sha(run / 'primary_ep99.log'),
            'protocol_sha256': sha(run / 'evaluation_protocol.json')}, indent=2)+'\n', encoding='utf-8')
        (run / "metrics.json").write_text(json.dumps(score_matrix(matrix), indent=2) + "\n", encoding="utf-8")
        (run / "PRIMARY_EVAL_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("Completed in %.1f seconds." % elapsed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "eval"))
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    p.add_argument("--tag", default="fixed_v1")
    p.add_argument("--scheduler-mode", choices=("batch_step",), default="batch_step")
    p.add_argument("--feature-root", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--python")
    p.add_argument("--gpu", default="0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    execute(p.parse_args())


if __name__ == "__main__":
    main()
