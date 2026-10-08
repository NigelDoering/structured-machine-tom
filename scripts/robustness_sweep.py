"""
Run robustness training sweeps serially.

The sweep is intentionally pre-registered in this script: both BDIBottleneck
and AuxContext use the same seeds, hyperparameter configs, training budget,
and belief-loss options. Checkpoints remain under outputs/ and are not meant
to be committed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diag_utils import resolve, write_text


MODELS = {
    "bdi": {
        "label": "BDIBottleneck",
        "script": "scripts/train_bdi_bottleneck.py",
        "out_name": "bdi_bottleneck",
    },
    "aux": {
        "label": "AuxSupervisedContextGoalPredictor",
        "script": "scripts/train_context_goal_predictor_aux.py",
        "out_name": "context_goal_aux",
    },
}

CONFIGS = {
    "default_weighted": {
        "lr": 3e-4,
        "belief_loss": "weighted_bce",
        "belief_weight": 5.0,
        "focal_gamma": 2.0,
        "balanced_max_weight": 12.0,
    },
    "low_lr_weighted": {
        "lr": 1e-4,
        "belief_loss": "weighted_bce",
        "belief_weight": 5.0,
        "focal_gamma": 2.0,
        "balanced_max_weight": 12.0,
    },
    "balanced_bce": {
        "lr": 3e-4,
        "belief_loss": "balanced_bce",
        "belief_weight": 5.0,
        "focal_gamma": 2.0,
        "balanced_max_weight": 12.0,
    },
    "balanced_focal": {
        "lr": 3e-4,
        "belief_loss": "balanced_focal",
        "belief_weight": 5.0,
        "focal_gamma": 2.0,
        "balanced_max_weight": 12.0,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair robustness training sweep.")
    p.add_argument("--phase", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--out-root", default="outputs/diagnostics/robustness")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["bdi", "aux"])
    p.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--chunk-epochs", type=int, default=0, help="Run training in resumable chunks; 0 disables chunking.")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _base_command(
    model_key: str,
    config_name: str,
    seed: int,
    args: argparse.Namespace,
    batch: int,
    no_cuda: bool,
) -> tuple[list[str], Path]:
    model = MODELS[model_key]
    config = CONFIGS[config_name]
    out_base = resolve(args.out_root) / args.phase / config_name / model["out_name"]
    cmd = [
        str(resolve(".venv/bin/python")),
        model["script"],
        "--epochs",
        str(args.epochs),
        "--batch",
        str(batch),
        "--K",
        str(args.K),
        "--K-recent",
        str(args.K_recent),
        "--lr",
        str(config["lr"]),
        "--belief-loss",
        str(config["belief_loss"]),
        "--belief-weight",
        str(config["belief_weight"]),
        "--focal-gamma",
        str(config["focal_gamma"]),
        "--balanced-max-weight",
        str(config["balanced_max_weight"]),
        "--splits",
        _rel(resolve(args.splits_dir)),
        "--out",
        _rel(out_base),
        "--seed",
        str(seed),
    ]
    state_path = out_base / f"seed_{seed}" / "last_state.pt"
    if args.chunk_epochs > 0:
        cmd += [
            "--resume-state",
            _rel(state_path),
            "--state-out",
            _rel(state_path),
            "--max-epochs-this-run",
            str(args.chunk_epochs),
        ]
    if no_cuda:
        cmd.append("--no-cuda")
    return cmd, out_base / f"seed_{seed}"


def _run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write("\n\n$ " + _quote_cmd(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nexit_code={proc.returncode}\n")
        return int(proc.returncode)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _completed(run_dir: Path) -> bool:
    return (run_dir / "accuracy_results.json").exists() and (run_dir / "best.pt").exists()


def _attempt_limit(args: argparse.Namespace) -> int:
    if args.chunk_epochs <= 0:
        return max(1, args.max_retries)
    chunks = math.ceil(args.epochs / max(1, args.chunk_epochs))
    return max(1, chunks + max(1, args.max_retries) + 2)


def _execute_run(model_key: str, config_name: str, seed: int, args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    batch_attempts = [args.batch]
    if args.batch > 16:
        batch_attempts.append(16)
    cpu_attempted = False
    start = time.time()
    last_code = 0
    last_cmd: list[str] = []
    run_dir: Path | None = None

    for batch in batch_attempts:
        for _ in range(_attempt_limit(args)):
            cmd, run_dir = _base_command(model_key, config_name, seed, args, batch=batch, no_cuda=args.no_cuda)
            last_cmd = cmd
            if _completed(run_dir):
                return {
                    "status": "completed_reused",
                    "exit_code": 0,
                    "batch": batch,
                    "device": "cpu" if args.no_cuda else "auto",
                    "command": _quote_cmd(cmd),
                    "run_dir": _rel(run_dir),
                    "duration_sec": round(time.time() - start, 3),
                }
            log_path = resolve(args.out_root) / "logs" / args.phase / config_name / model_key / f"seed_{seed}.log"
            last_code = _run_command(cmd, log_path, env)
            if last_code == 0 and _completed(run_dir):
                return {
                    "status": "completed",
                    "exit_code": 0,
                    "batch": batch,
                    "device": "cpu" if args.no_cuda else "auto",
                    "command": _quote_cmd(cmd),
                    "run_dir": _rel(run_dir),
                    "duration_sec": round(time.time() - start, 3),
                }
            if last_code != 0:
                break
            if args.chunk_epochs <= 0:
                break

    if not args.no_cuda:
        cpu_attempted = True
        cmd, run_dir = _base_command(model_key, config_name, seed, args, batch=min(args.batch, 16), no_cuda=True)
        last_cmd = cmd
        log_path = resolve(args.out_root) / "logs" / args.phase / config_name / model_key / f"seed_{seed}_cpu.log"
        for _ in range(_attempt_limit(args)):
            if _completed(run_dir):
                return {
                    "status": "completed_cpu_reused",
                    "exit_code": 0,
                    "batch": min(args.batch, 16),
                    "device": "cpu",
                    "command": _quote_cmd(cmd),
                    "run_dir": _rel(run_dir),
                    "duration_sec": round(time.time() - start, 3),
                }
            last_code = _run_command(cmd, log_path, env)
            if last_code == 0 and run_dir is not None and _completed(run_dir):
                return {
                    "status": "completed_cpu_fallback",
                    "exit_code": 0,
                    "batch": min(args.batch, 16),
                    "device": "cpu",
                    "command": _quote_cmd(cmd),
                    "run_dir": _rel(run_dir),
                    "duration_sec": round(time.time() - start, 3),
                }
            if last_code != 0:
                break
            if args.chunk_epochs <= 0:
                break

    return {
        "status": "failed",
        "exit_code": last_code,
        "batch": "",
        "device": "cpu" if cpu_attempted else ("cpu" if args.no_cuda else "auto"),
        "command": _quote_cmd(last_cmd),
        "run_dir": _rel(run_dir) if run_dir is not None else "",
        "duration_sec": round(time.time() - start, 3),
    }


def main() -> None:
    args = parse_args()
    out_root = resolve(args.out_root)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("BDI_FORCE_MATH_SDP", "1")
    env.setdefault("BDI_DISABLE_FOREACH_OPT", "1")

    rows: list[dict[str, Any]] = []
    commands: list[str] = ["# Robustness Sweep Commands", ""]
    for config_name in args.configs:
        for model_key in args.models:
            for seed in args.seeds:
                cmd, run_dir = _base_command(model_key, config_name, seed, args, batch=args.batch, no_cuda=args.no_cuda)
                base_row = {
                    "phase": args.phase,
                    "config": config_name,
                    "model_key": model_key,
                    "model": MODELS[model_key]["label"],
                    "seed": seed,
                    "epochs": args.epochs,
                    "splits_dir": _rel(resolve(args.splits_dir)),
                    "run_dir": _rel(run_dir),
                    "planned_command": _quote_cmd(cmd),
                }
                commands.append(f"- `{_quote_cmd(cmd)}`")
                if args.dry_run:
                    rows.append({**base_row, "status": "planned", "exit_code": ""})
                else:
                    result = _execute_run(model_key, config_name, seed, args, env)
                    rows.append({**base_row, **result})
                    _write_manifest(out_root / args.phase / "manifest.csv", rows)
                    if result["status"] == "failed":
                        break
            if rows and rows[-1].get("status") == "failed":
                break
        if rows and rows[-1].get("status") == "failed":
            break

    _write_manifest(out_root / args.phase / "manifest.csv", rows)
    write_text(out_root / args.phase / "RUN_COMMANDS.md", "\n".join(commands) + "\n")
    summary = {
        "phase": args.phase,
        "dry_run": args.dry_run,
        "planned_or_run_rows": len(rows),
        "status_counts": {status: sum(1 for r in rows if r.get("status") == status) for status in sorted({str(r.get("status")) for r in rows})},
    }
    (out_root / args.phase / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if any(row.get("status") == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
