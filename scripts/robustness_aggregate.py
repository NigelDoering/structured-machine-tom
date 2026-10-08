"""
Aggregate robustness sweep checkpoints.

This evaluates completed BDIBottleneck and AuxContext checkpoints with the
same split settings, then summarizes seed/config robustness and selects the
single belief-optimized config for dense confirmation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from eval_aux_baseline import _aux_epoch_metrics, _bdi_epoch_metrics
from diag_utils import (
    get_device,
    load_aux_context,
    load_bdi_bottleneck,
    load_split_records,
    node_coords_from_graph,
    poi_metadata,
    read_csv_dicts,
    resolve,
    safe_mean,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_CONFIG = "default_weighted"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate robustness sweep results.")
    p.add_argument("--robustness-root", default="outputs/diagnostics/robustness")
    p.add_argument("--phase", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--splits", nargs="+", default=["val", "test", "test_new_agent", "test_bh", "test_new_agent_bh"])
    p.add_argument("--out-dir", default="")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection-goal-tolerance", type=float, default=0.02)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=48)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _manifest_rows(root: Path, phase: str) -> list[dict[str, str]]:
    path = root / phase / "manifest.csv"
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    return read_csv_dicts(path)


def _checkpoint_val_metrics(ckpt_path: Path) -> dict[str, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("val_metrics", {})
    return {
        "val_goal_top1_ckpt": float(metrics.get("goal_top1", float("nan"))),
        "val_belief_f1_ckpt": float(metrics.get("belief_f1", float("nan"))),
        "epoch": int(ckpt.get("epoch", -1)),
    }


def _std(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _aggregate(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, vals in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        row["n"] = len(vals)
        for metric in metrics:
            numbers = [float(v[metric]) for v in vals if v.get(metric, "") != ""]
            row[f"{metric}_mean"] = safe_mean(numbers)
            row[f"{metric}_std"] = _std(numbers)
        out.append(row)
    return out


def _select_dense_config(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    val_rows = [
        r
        for r in rows
        if r["model_key"] == "bdi" and r["split"] == "val" and r["status"].startswith("completed")
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in val_rows:
        grouped[row["config"]].append(row)
    if DEFAULT_CONFIG not in grouped:
        return {"selected_config": "", "selection_reason": "default_missing"}
    default_goal = safe_mean([float(r["top1_full"]) for r in grouped[DEFAULT_CONFIG]])
    candidates = []
    for config, vals in grouped.items():
        if config == DEFAULT_CONFIG:
            continue
        goal = safe_mean([float(r["top1_full"]) for r in vals])
        f1 = safe_mean([float(r["closed_f1"]) for r in vals])
        candidates.append({"config": config, "goal": goal, "closed_f1": f1, "within_tolerance": goal >= default_goal - tolerance})
    if not candidates:
        return {"selected_config": "", "selection_reason": "no_non_default_candidates", "default_goal": default_goal}
    eligible = [c for c in candidates if c["within_tolerance"]]
    pool = eligible if eligible else candidates
    selected = sorted(pool, key=lambda c: (c["closed_f1"], c["goal"], c["config"]), reverse=True)[0]
    return {
        "selected_config": selected["config"],
        "selection_reason": "within_goal_tolerance" if eligible else "highest_f1_no_goal_tolerance_candidate",
        "default_goal_top1": default_goal,
        "goal_tolerance": tolerance,
        "candidates": candidates,
    }


def _dense_selection_metadata(robustness_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sparse_selection_path = robustness_root / "sparse" / "aggregate" / "selected_dense_config.json"
    if sparse_selection_path.exists():
        with sparse_selection_path.open("r", encoding="utf-8") as f:
            selection = json.load(f)
        selection["selection_source"] = str(sparse_selection_path)
        return selection
    configs = sorted({r["config"] for r in rows if r.get("config") != DEFAULT_CONFIG})
    return {
        "selected_config": ",".join(configs),
        "selection_reason": "dense_phase_non_default_configs" if configs else "dense_phase_default_only",
    }


def _summary(by_config: list[dict[str, Any]], paired: list[dict[str, Any]], selection: dict[str, Any]) -> str:
    lines = [
        "# Robustness Summary",
        "",
        f"- Dense confirmation selected config: `{selection.get('selected_config', '')}` ({selection.get('selection_reason', '')}).",
        "",
        "## Key Config Means",
        "",
    ]
    for row in by_config:
        if row.get("split") != "test":
            continue
        lines.append(
            f"- {row['model']} / {row['config']} / test: "
            f"top1={float(row['top1_full_mean']):.4f}±{float(row['top1_full_std']):.4f}, "
            f"closed_f1={float(row['closed_f1_mean']):.4f}±{float(row['closed_f1_std']):.4f}."
        )
    if paired:
        lines += ["", "## Paired BDIBottleneck minus AuxContext", ""]
        for row in paired:
            if row.get("split") == "test":
                lines.append(
                    f"- {row['config']} / test: Δtop1={float(row['delta_top1_full_mean']):.4f}±{float(row['delta_top1_full_std']):.4f}, "
                    f"Δclosed_f1={float(row['delta_closed_f1_mean']):.4f}±{float(row['delta_closed_f1_std']):.4f}."
                )
    lines += [
        "",
        "This summary is diagnostic and reports same-budget robustness, not tuned winner selection.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    robustness_root = resolve(args.robustness_root)
    out_dir = resolve(args.out_dir) if args.out_dir else robustness_root / args.phase / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.no_cuda)
    meta = poi_metadata()
    node_coords = node_coords_from_graph()
    records_by_split = load_split_records(args.splits_dir, args.splits, smoke=args.smoke, smoke_n=args.smoke_n)
    if not records_by_split:
        raise SystemExit(f"No requested split files found under {resolve(args.splits_dir)}")

    manifest = _manifest_rows(robustness_root, args.phase)
    metric_rows: list[dict[str, Any]] = []
    for row in manifest:
        status = row.get("status", "")
        if not status.startswith("completed"):
            continue
        run_dir = resolve(row["run_dir"])
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            continue
        model_key = row["model_key"]
        if model_key == "bdi":
            model, _ = load_bdi_bottleneck(ckpt_path, device)
            eval_fn = _bdi_epoch_metrics
        elif model_key == "aux":
            model, _ = load_aux_context(ckpt_path, device)
            eval_fn = _aux_epoch_metrics
        else:
            continue
        ckpt_metrics = _checkpoint_val_metrics(ckpt_path)
        for split, records in records_by_split.items():
            metrics = eval_fn(model, records, node_coords, meta["poi_to_idx"], args, device, split)
            metric_rows.append(
                {
                    "phase": args.phase,
                    "config": row["config"],
                    "model_key": model_key,
                    "model": row["model"],
                    "seed": int(row["seed"]),
                    "split": split,
                    "status": status,
                    "run_dir": row["run_dir"],
                    **ckpt_metrics,
                    **metrics["goal"],
                    **{k: v for k, v in metrics["belief"].items() if k not in {"model", "split"}},
                    **{k: v for k, v in metrics["desire"].items() if k not in {"model", "split"}},
                }
            )

    metrics = [
        "top1_full",
        "top5_full",
        "nll_full",
        "closed_f1",
        "closed_precision",
        "closed_recall",
        "pr_auc_closed",
        "desire_kl",
        "val_goal_top1_ckpt",
        "val_belief_f1_ckpt",
    ]
    by_config = _aggregate(metric_rows, ["phase", "config", "model_key", "model", "split"], metrics)

    paired_rows: list[dict[str, Any]] = []
    lookup = {
        (r["config"], r["seed"], r["split"], r["model_key"]): r
        for r in metric_rows
    }
    for row in metric_rows:
        if row["model_key"] != "bdi":
            continue
        aux = lookup.get((row["config"], row["seed"], row["split"], "aux"))
        if not aux:
            continue
        paired_rows.append(
            {
                "phase": row["phase"],
                "config": row["config"],
                "seed": row["seed"],
                "split": row["split"],
                "delta_top1_full": float(row["top1_full"]) - float(aux["top1_full"]),
                "delta_top5_full": float(row["top5_full"]) - float(aux["top5_full"]),
                "delta_closed_f1": float(row["closed_f1"]) - float(aux["closed_f1"]),
                "delta_pr_auc_closed": float(row["pr_auc_closed"]) - float(aux["pr_auc_closed"]),
            }
        )
    paired = _aggregate(
        paired_rows,
        ["phase", "config", "split"],
        ["delta_top1_full", "delta_top5_full", "delta_closed_f1", "delta_pr_auc_closed"],
    )
    selection = (
        _select_dense_config(metric_rows, args.selection_goal_tolerance)
        if args.phase == "sparse"
        else _dense_selection_metadata(robustness_root, metric_rows)
    )

    write_csv(out_dir / "robustness_run_metrics.csv", metric_rows)
    write_csv(out_dir / "robustness_by_config.csv", by_config)
    write_csv(out_dir / "robustness_paired_differences.csv", paired)
    write_json(out_dir / "selected_dense_config.json", selection)
    write_text(out_dir / "summary.md", _summary(by_config, paired, selection))
    print(f"Robustness aggregate written to {out_dir}")


if __name__ == "__main__":
    main()
