"""
Bootstrap confidence intervals for the trajectory-level counterfactual diagnostics.

Intervals are clustered by agent_id when available and by record_index
otherwise. The script consumes generated counterfactual CSVs and writes
lightweight CSV/Markdown summaries only.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diag_utils import read_csv_dicts, resolve, safe_mean, write_csv, write_text


DEFAULT_INPUTS = [
    ("true_counterfactual_rollouts", "rollout", "outputs/diagnostics/analyses/true_counterfactual_rollouts"),
    ("dense_true_counterfactual_rollouts", "rollout", "outputs/diagnostics/analyses/dense_true_counterfactual_rollouts"),
    ("model_planned_counterfactual_paths", "model_planned", "outputs/diagnostics/analyses/model_planned_counterfactual_paths"),
    ("dense_model_planned_counterfactual_paths", "model_planned", "outputs/diagnostics/analyses/dense_model_planned_counterfactual_paths"),
    ("robustness_model_planned_sparse", "model_planned_tree", "outputs/diagnostics/analyses/robustness_model_planned_counterfactual_paths/sparse"),
    ("robustness_model_planned_dense", "model_planned_tree", "outputs/diagnostics/analyses/robustness_model_planned_counterfactual_paths/dense"),
]

ROLLOUT_METRICS = ["top1_correct", "top5_correct", "prob_counterfactual_goal"]
MODEL_PLANNED_GOAL_METRICS = ["goal_changed_vs_original", "goal_changed_vs_normal", "cf_prob_original_goal", "cf_prob_top1_goal"]
MODEL_PLANNED_PATH_METRICS = ["planned_success", "node_overlap_original", "edge_overlap_original", "route_length_ratio_original"]
MODEL_PLANNED_SIM_METRICS = [
    "simulator_success",
    "simulator_goal_top1_match",
    "simulator_goal_top5_match",
    "cf_prob_simulator_goal",
    "node_overlap_simulator",
    "edge_overlap_simulator",
    "route_length_ratio_simulator",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Counterfactual bootstrap CIs.")
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/counterfactual_bootstrap_ci")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260705)
    p.add_argument("--input", action="append", default=[], help="Optional input triple: stage,kind,path")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _stable_offset(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _cluster_key(row: dict[str, Any]) -> str:
    if row.get("agent_id"):
        return f"agent:{row['agent_id']}"
    if row.get("record_index"):
        return f"record:{row['record_index']}"
    return f"row:{id(row)}"


def _bootstrap_ci(rows: list[dict[str, Any]], metric: str, samples: int, seed: int) -> dict[str, Any]:
    values = [_float(row.get(metric, "")) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"n": 0, "clusters": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get(metric, ""))
        if math.isfinite(value):
            clusters[_cluster_key(row)].append(value)
    cluster_keys = sorted(clusters)
    cluster_values = [clusters[key] for key in cluster_keys]
    cluster_sums = np.asarray([sum(vals) for vals in cluster_values], dtype=np.float64)
    cluster_counts = np.asarray([len(vals) for vals in cluster_values], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_chunks = []
    remaining = int(samples)
    chunk_size = 1000
    while remaining > 0:
        n_chunk = min(chunk_size, remaining)
        sampled = rng.integers(0, len(cluster_values), size=(n_chunk, len(cluster_values)))
        sums = cluster_sums[sampled].sum(axis=1)
        counts = cluster_counts[sampled].sum(axis=1)
        boot_chunks.append(sums / counts)
        remaining -= n_chunk
    boot = np.concatenate(boot_chunks) if boot_chunks else np.asarray([], dtype=np.float64)
    return {
        "n": len(finite),
        "clusters": len(cluster_values),
        "mean": float(np.mean(finite)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def _group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def _ci_rows(
    stage: str,
    source: str,
    rows: list[dict[str, Any]],
    group_keys: list[str],
    metrics: list[str],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    out = []
    for group_key, vals in sorted(_group_rows(rows, group_keys).items()):
        group = {key: value for key, value in zip(group_keys, group_key)}
        for metric in metrics:
            stat = _bootstrap_ci(
                vals,
                metric,
                samples=samples,
                seed=seed + _stable_offset(stage, source, metric, *[str(x) for x in group_key]),
            )
            if stat["n"] <= 0:
                continue
            out.append(
                {
                    "stage": stage,
                    "source": source,
                    "metric": metric,
                    **group,
                    **stat,
                    "bootstrap_samples": samples,
                }
            )
    return out


def _parse_inputs(inputs: list[str]) -> list[tuple[str, str, str]]:
    if not inputs:
        return DEFAULT_INPUTS
    parsed = []
    for spec in inputs:
        parts = spec.split(",", 2)
        if len(parts) != 3:
            raise SystemExit(f"Invalid --input spec, expected stage,kind,path: {spec}")
        parsed.append((parts[0], parts[1], parts[2]))
    return parsed


def _load_stage(stage: str, kind: str, path: Path, samples: int, seed: int) -> list[dict[str, Any]]:
    if kind == "rollout":
        rows = []
        for cf_type in ("belief", "desire"):
            file_path = path / f"cf_rollout_{cf_type}.csv"
            if file_path.exists():
                rows.extend(read_csv_dicts(file_path))
        return _ci_rows(
            stage,
            "simulator_resampled_rollout",
            rows,
            ["split", "cf_type", "fraction"],
            ROLLOUT_METRICS,
            samples,
            seed,
        )
    if kind == "model_planned":
        out = []
        goal_path = path / "goal_changes.csv"
        if goal_path.exists():
            out.extend(
                _ci_rows(
                    stage,
                    "model_planned_goal",
                    read_csv_dicts(goal_path),
                    ["split", "cf_type", "channel_mode", "fraction"],
                    MODEL_PLANNED_GOAL_METRICS,
                    samples,
                    seed,
                )
            )
        path_path = path / "planned_paths.csv"
        if path_path.exists():
            out.extend(
                _ci_rows(
                    stage,
                    "model_planned_path",
                    read_csv_dicts(path_path),
                    ["split", "cf_type", "channel_mode", "fraction"],
                    MODEL_PLANNED_PATH_METRICS,
                    samples,
                    seed,
                )
            )
        sim_path = path / "model_vs_simulator.csv"
        if sim_path.exists():
            out.extend(
                _ci_rows(
                    stage,
                    "model_vs_simulator",
                    read_csv_dicts(sim_path),
                    ["split", "cf_type", "channel_mode", "fraction"],
                    MODEL_PLANNED_SIM_METRICS,
                    samples,
                    seed,
                )
            )
        return out
    if kind == "model_planned_tree":
        goal_rows = []
        path_rows = []
        sim_rows = []
        for config_dir in sorted(p for p in path.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
                common = {"train_config": config_dir.name, "seed": seed_dir.name.removeprefix("seed_")}
                goal_path = seed_dir / "goal_changes.csv"
                if goal_path.exists():
                    goal_rows.extend({**common, **row} for row in read_csv_dicts(goal_path))
                path_path = seed_dir / "planned_paths.csv"
                if path_path.exists():
                    path_rows.extend({**common, **row} for row in read_csv_dicts(path_path))
                sim_path = seed_dir / "model_vs_simulator.csv"
                if sim_path.exists():
                    sim_rows.extend({**common, **row} for row in read_csv_dicts(sim_path))
        out = []
        if goal_rows:
            out.extend(
                _ci_rows(
                    stage,
                    "model_planned_goal",
                    goal_rows,
                    ["train_config", "split", "cf_type", "channel_mode", "fraction"],
                    MODEL_PLANNED_GOAL_METRICS,
                    samples,
                    seed,
                )
            )
        if path_rows:
            out.extend(
                _ci_rows(
                    stage,
                    "model_planned_path",
                    path_rows,
                    ["train_config", "split", "cf_type", "channel_mode", "fraction"],
                    MODEL_PLANNED_PATH_METRICS,
                    samples,
                    seed,
                )
            )
        if sim_rows:
            out.extend(
                _ci_rows(
                    stage,
                    "model_vs_simulator",
                    sim_rows,
                    ["train_config", "split", "cf_type", "channel_mode", "fraction"],
                    MODEL_PLANNED_SIM_METRICS,
                    samples,
                    seed,
                )
            )
        return out
    raise SystemExit(f"Unknown input kind: {kind}")


def _summary(rows: list[dict[str, Any]]) -> str:
    stage_count = len({row["stage"] for row in rows})
    metrics = sorted({row["metric"] for row in rows})
    lines = [
        "# Counterfactual Bootstrap CI Summary",
        "",
        f"- Stages summarized: {stage_count}.",
        f"- CI rows: {len(rows)}.",
        f"- Metrics: {', '.join(metrics)}.",
        "",
        "Intervals are 95% percentile bootstrap intervals clustered by `agent_id` when available and `record_index` otherwise.",
        "",
        "## Selected Overall Rows",
        "",
    ]
    selected = [
        row
        for row in rows
        if row.get("split") == "test_new_agent"
        and row.get("fraction") in {"0.75", "0.7500000000000000", 0.75}
        and row["metric"] in {"top1_correct", "simulator_goal_top1_match", "goal_changed_vs_original"}
    ]
    for row in selected[:24]:
        lines.append(
            f"- {row['stage']} / {row['source']} / {row['metric']} / {row.get('cf_type', '')} / "
            f"{row.get('channel_mode', '')}: mean={float(row['mean']):.4f}, "
            f"95% CI=[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    samples = 50 if args.smoke else args.bootstrap_samples
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for stage, kind, path in _parse_inputs(args.input):
        stage_path = resolve(path)
        if not stage_path.exists():
            continue
        rows.extend(_load_stage(stage, kind, stage_path, samples, args.seed))
    write_csv(out_dir / "counterfactual_ci.csv", rows)
    write_text(out_dir / "summary.md", _summary(rows))
    print(f"Counterfactual bootstrap CIs written to {out_dir} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
