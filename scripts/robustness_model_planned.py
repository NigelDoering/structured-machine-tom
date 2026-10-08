"""
Run and aggregate model-planned counterfactual paths for robustness checkpoints.

This is intentionally limited to BDIBottleneck checkpoints, because the
model-planned path diagnostic intervenes on the explicit bottleneck channels.
It compares default training against the pre-registered selected belief-loss
configuration without retraining or tuning during evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diag_utils import read_csv_dicts, resolve, safe_mean, write_csv, write_text


METRICS = [
    "planned_success",
    "goal_changed_vs_original",
    "goal_changed_vs_normal",
    "node_overlap_original",
    "edge_overlap_original",
    "route_length_ratio_original",
    "simulator_success",
    "simulator_goal_top1_match",
    "simulator_goal_top5_match",
    "cf_prob_simulator_goal",
    "node_overlap_simulator",
    "edge_overlap_simulator",
    "route_length_ratio_simulator",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness model-planned path runner.")
    p.add_argument("--phase", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--robustness-root", default="outputs/diagnostics/robustness")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--out-root", default="outputs/diagnostics/analyses/robustness_model_planned_counterfactual_paths")
    p.add_argument("--configs", nargs="+", default=["default_weighted", "selected"])
    p.add_argument("--selected-config-json", default="outputs/diagnostics/robustness/sparse/aggregate/selected_dense_config.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--splits", nargs="+", default=["test", "test_new_agent"])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--router-k", type=int, default=1)
    p.add_argument("--router-temperature", type=float, default=100.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--record-limit", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _selected_config(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Selected config JSON not found: {path}")
    data = json.loads(path.read_text())
    config = data.get("selected_config") or data.get("config")
    if not config:
        raise SystemExit(f"Selected config JSON does not contain selected_config/config: {path}")
    return str(config)


def _configs(args: argparse.Namespace) -> list[str]:
    selected = _selected_config(resolve(args.selected_config_json)) if "selected" in args.configs else ""
    out: list[str] = []
    for config in args.configs:
        value = selected if config == "selected" else config
        if value and value not in out:
            out.append(value)
    return out


def _run_dir(args: argparse.Namespace, config: str, seed: int) -> Path:
    return resolve(args.out_root) / args.phase / config / f"seed_{seed}"


def _ckpt_path(args: argparse.Namespace, config: str, seed: int) -> Path:
    return resolve(args.robustness_root) / args.phase / config / "bdi_bottleneck" / f"seed_{seed}" / "best.pt"


def _command(
    args: argparse.Namespace,
    config: str,
    seed: int,
    *,
    batch: int | None = None,
    no_cuda: bool | None = None,
) -> list[str]:
    use_no_cuda = args.no_cuda if no_cuda is None else no_cuda
    cmd = [
        str(resolve(".venv/bin/python")),
        "scripts/cf_model_planned_paths.py",
        "--bnk-ckpt",
        _rel(_ckpt_path(args, config, seed)),
        "--splits-dir",
        _rel(resolve(args.splits_dir)),
        "--out-dir",
        _rel(_run_dir(args, config, seed)),
        "--splits",
        *args.splits,
        "--fractions",
        *[str(x) for x in args.fractions],
        "--router-k",
        str(args.router_k),
        "--router-temperature",
        str(args.router_temperature),
        "--batch",
        str(batch if batch is not None else args.batch),
        "--chunk-label",
        f"{args.phase}_{config}_seed_{seed}",
    ]
    if args.record_limit > 0:
        cmd += ["--record-limit", str(args.record_limit)]
    if args.smoke:
        cmd.append("--smoke")
    if use_no_cuda:
        cmd.append("--no-cuda")
    return cmd


def _completed(path: Path) -> bool:
    return (path / "goal_changes.csv").exists() and (path / "model_vs_simulator.csv").exists() and (path / "summary.md").exists()


def _run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write("\n\n$ " + _quote_cmd(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nexit_code={proc.returncode}\n")
        return int(proc.returncode)


def _attempt_commands(args: argparse.Namespace, config: str, seed: int) -> list[list[str]]:
    commands = [_command(args, config, seed)]
    retry_batch = min(args.batch, 16)
    if retry_batch != args.batch:
        commands.append(_command(args, config, seed, batch=retry_batch))
    if not args.no_cuda:
        commands.append(_command(args, config, seed, batch=retry_batch, no_cuda=True))
    out: list[list[str]] = []
    seen: set[str] = set()
    for cmd in commands:
        text = _quote_cmd(cmd)
        if text not in seen:
            seen.add(text)
            out.append(cmd)
    return out


def _run_with_retries(args: argparse.Namespace, config: str, seed: int, log_path: Path, env: dict[str, str]) -> tuple[int, str]:
    out_dir = _run_dir(args, config, seed)
    last_code = 1
    attempts: list[str] = []
    if out_dir.exists() and not _completed(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    for idx, cmd in enumerate(_attempt_commands(args, config, seed), start=1):
        if idx > 1 and out_dir.exists() and not _completed(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        code = _run_command(cmd, log_path, env)
        last_code = code
        attempts.append(f"attempt{idx}:exit={code}:{_quote_cmd(cmd)}")
        if code == 0 and _completed(out_dir):
            return 0, " | ".join(attempts)
    return last_code, " | ".join(attempts)


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def _aggregate_run(args: argparse.Namespace, config: str, seed: int) -> list[dict[str, Any]]:
    out_dir = _run_dir(args, config, seed)
    rows = read_csv_dicts(out_dir / "model_vs_simulator.csv")
    path_rows = read_csv_dicts(out_dir / "planned_paths.csv")
    path_lookup: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in path_rows:
        key = (
            row.get("split", ""),
            row.get("cf_type", ""),
            row.get("channel_mode", ""),
            row.get("fraction", ""),
            row.get("record_index", ""),
        )
        path_lookup[key] = row

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("split", ""),
            row.get("cf_type", ""),
            row.get("channel_mode", ""),
            row.get("fraction", ""),
            row.get("record_index", ""),
        )
        merged = {**path_lookup.get(key, {}), **row}
        grouped[(row.get("split", ""), row.get("cf_type", ""), row.get("channel_mode", ""), row.get("fraction", ""))].append(merged)

    out: list[dict[str, Any]] = []
    for (split, cf_type, channel_mode, fraction), vals in sorted(grouped.items()):
        item: dict[str, Any] = {
            "phase": args.phase,
            "config": config,
            "seed": seed,
            "split": split,
            "cf_type": cf_type,
            "channel_mode": channel_mode,
            "fraction": fraction,
            "n": len(vals),
        }
        for metric in METRICS:
            item[metric] = safe_mean([_float(row, metric) for row in vals])
        out.append(item)
    return out


def _write_run_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
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


def _by_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["phase"]),
                str(row["config"]),
                str(row["split"]),
                str(row["cf_type"]),
                str(row["channel_mode"]),
                str(row["fraction"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (phase, config, split, cf_type, channel_mode, fraction), vals in sorted(grouped.items()):
        item: dict[str, Any] = {
            "phase": phase,
            "config": config,
            "split": split,
            "cf_type": cf_type,
            "channel_mode": channel_mode,
            "fraction": fraction,
            "seed_count": len({int(row["seed"]) for row in vals}),
        }
        for metric in METRICS:
            item[f"{metric}_mean"] = safe_mean([_float(row, metric) for row in vals])
        out.append(item)
    return out


def _summary(rows: list[dict[str, Any]], by_config: list[dict[str, Any]], configs: list[str]) -> str:
    lines = [
        "# Robustness Model-Planned Counterfactual Path Summary",
        "",
        f"- Configs: {', '.join(configs)}.",
        f"- Per-seed metric rows: {len(rows)}.",
        f"- Config aggregate rows: {len(by_config)}.",
        "",
        "## Overall By Config",
        "",
    ]
    overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        overall[str(row["config"])].append(row)
    for config in configs:
        vals = overall.get(config, [])
        if not vals:
            lines.append(f"- `{config}`: no completed rows.")
            continue
        lines.append(
            f"- `{config}`: simulator top-1={safe_mean([_float(r, 'simulator_goal_top1_match') for r in vals]):.4f}, "
            f"simulator top-5={safe_mean([_float(r, 'simulator_goal_top5_match') for r in vals]):.4f}, "
            f"goal-change vs original={safe_mean([_float(r, 'goal_changed_vs_original') for r in vals]):.4f}, "
            f"node-overlap original={safe_mean([_float(r, 'node_overlap_original') for r in vals]):.4f}."
        )
    lines += [
        "",
        "These rows use frozen BDIBottleneck checkpoints only. AuxContext is excluded because it does not expose the same explicit bottleneck intervention interface.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    configs = _configs(args)
    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("BDI_FORCE_MATH_SDP", "1")

    manifest: list[dict[str, Any]] = []
    commands: list[str] = ["# Robustness Model-Planned Commands", ""]
    if not args.aggregate_only:
        for config in configs:
            for seed in args.seeds:
                out_dir = _run_dir(args, config, seed)
                ckpt = _ckpt_path(args, config, seed)
                cmd = _command(args, config, seed)
                commands.append(f"- `{_quote_cmd(cmd)}`")
                row: dict[str, Any] = {
                    "phase": args.phase,
                    "config": config,
                    "seed": seed,
                    "checkpoint": _rel(ckpt),
                    "out_dir": _rel(out_dir),
                    "command": _quote_cmd(cmd),
                }
                if not ckpt.exists():
                    manifest.append({**row, "status": "missing_checkpoint", "exit_code": ""})
                    continue
                if _completed(out_dir):
                    manifest.append({**row, "status": "completed_reused", "exit_code": 0})
                    continue
                if args.dry_run:
                    manifest.append({**row, "status": "planned", "exit_code": ""})
                    continue
                log_path = out_root / "logs" / args.phase / config / f"seed_{seed}.log"
                code, attempts = _run_with_retries(args, config, seed, log_path, env)
                status = "completed" if code == 0 and _completed(out_dir) else "failed"
                manifest.append({**row, "status": status, "exit_code": code, "attempts": attempts})
                _write_run_manifest(out_root / args.phase / "manifest.csv", manifest)
                if status == "failed":
                    break
            if manifest and manifest[-1].get("status") == "failed":
                break

    metric_rows: list[dict[str, Any]] = []
    for config in configs:
        for seed in args.seeds:
            if _completed(_run_dir(args, config, seed)):
                metric_rows.extend(_aggregate_run(args, config, seed))
    by_config = _by_config(metric_rows)
    write_text(out_root / args.phase / "RUN_COMMANDS.md", "\n".join(commands) + "\n")
    if manifest:
        _write_run_manifest(out_root / args.phase / "manifest.csv", manifest)
    write_csv(out_root / args.phase / "robustness_model_planned_metrics.csv", metric_rows)
    write_csv(out_root / args.phase / "robustness_model_planned_by_config.csv", by_config)
    write_text(out_root / args.phase / "summary.md", _summary(metric_rows, by_config, configs))
    print(json.dumps({"phase": args.phase, "configs": configs, "metric_rows": len(metric_rows)}, indent=2))
    if manifest and any(row.get("status") == "failed" for row in manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
