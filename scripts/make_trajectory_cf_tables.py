"""
Paper-facing trajectory-level counterfactual summaries.

This script converts the trajectory counterfactual artifacts
into lightweight tables that use the paper's probability-delta and top-k
metrics. It intentionally avoids reporting binary goal-change, node-overlap, or
edge-overlap metrics in the generated manuscript tables.
"""
from __future__ import annotations

import argparse
import csv
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

from diag_utils import read_csv_dicts, resolve, write_csv, write_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-facing trajectory counterfactual delta summaries.")
    p.add_argument("--out-dir", default="results/trajectory_counterfactuals")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--permutation-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stable_offset(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _cluster_key(row: dict[str, Any]) -> str:
    seed = row.get("seed", "")
    agent = row.get("agent_id", "")
    if seed != "" and agent != "":
        return f"seed_agent:{seed}:{agent}"
    if agent:
        return f"agent:{agent}"
    if row.get("record_index") not in {"", None}:
        return f"record:{row['record_index']}"
    return f"row:{id(row)}"


def _bootstrap_ci(rows: list[dict[str, Any]], metric: str, samples: int, seed: int) -> dict[str, Any]:
    finite_rows = [row for row in rows if math.isfinite(_float(row.get(metric, "")))]
    if not finite_rows:
        return {"n": 0, "clusters": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    clusters: dict[str, list[float]] = defaultdict(list)
    for row in finite_rows:
        clusters[_cluster_key(row)].append(_float(row[metric]))
    cluster_keys = sorted(clusters)
    cluster_sums = np.asarray([sum(clusters[key]) for key in cluster_keys], dtype=np.float64)
    cluster_counts = np.asarray([len(clusters[key]) for key in cluster_keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = []
    remaining = int(samples)
    chunk_size = 1000
    while remaining > 0:
        n_chunk = min(chunk_size, remaining)
        sampled = rng.integers(0, len(cluster_keys), size=(n_chunk, len(cluster_keys)))
        boot.append(cluster_sums[sampled].sum(axis=1) / cluster_counts[sampled].sum(axis=1))
        remaining -= n_chunk
    boot_arr = np.concatenate(boot)
    values = [_float(row[metric]) for row in finite_rows]
    return {
        "n": len(values),
        "clusters": len(cluster_keys),
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(boot_arr, 2.5)),
        "ci_high": float(np.percentile(boot_arr, 97.5)),
    }


def _cluster_metric_values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get(metric, ""))
        if math.isfinite(value):
            clusters[_cluster_key(row)].append(value)
    if not clusters:
        return np.asarray([], dtype=np.float64)
    return np.asarray([np.mean(clusters[key]) for key in sorted(clusters)], dtype=np.float64)


def _random_sign_pvalue(
    rows: list[dict[str, Any]],
    metric: str,
    alternative: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    values = _cluster_metric_values(rows, metric)
    if values.size == 0:
        return {
            "cluster_mean": float("nan"),
            "p_value": float("nan"),
            "permutation_samples": samples,
            "alternative": alternative,
        }

    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    count = 0
    remaining = int(samples)
    chunk_size = 1000
    while remaining > 0:
        n_chunk = min(chunk_size, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_chunk, values.size))
        permuted = (signs * values).mean(axis=1)
        if alternative == "less":
            count += int(np.count_nonzero(permuted <= observed))
        elif alternative == "greater":
            count += int(np.count_nonzero(permuted >= observed))
        elif alternative == "two-sided":
            count += int(np.count_nonzero(np.abs(permuted) >= abs(observed)))
        else:
            raise ValueError(f"Unsupported alternative: {alternative}")
        remaining -= n_chunk

    return {
        "cluster_mean": observed,
        "p_value": (count + 1.0) / (samples + 1.0),
        "permutation_samples": samples,
        "alternative": alternative,
    }


def _holm_adjust(rows: list[dict[str, Any]], p_key: str = "p_value", out_key: str = "p_adjusted") -> list[dict[str, Any]]:
    indexed = [(idx, _float(row.get(p_key))) for idx, row in enumerate(rows)]
    finite = sorted((item for item in indexed if math.isfinite(item[1])), key=lambda item: item[1])
    adjusted = [float("nan")] * len(rows)
    running = 0.0
    m = len(finite)
    for rank, (idx, p_value) in enumerate(finite):
        adj = min(1.0, (m - rank) * p_value)
        running = max(running, adj)
        adjusted[idx] = running
    return [{**row, out_key: adjusted[idx]} for idx, row in enumerate(rows)]


def _group(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return grouped


def _write_csv_lf(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ci_by_group(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    metrics: list[str],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    out = []
    for group_key, vals in sorted(_group(rows, group_keys).items()):
        group = dict(zip(group_keys, group_key))
        for metric in metrics:
            stat = _bootstrap_ci(
                vals,
                metric,
                samples,
                seed + _stable_offset(metric, *group_key),
            )
            if stat["n"] == 0:
                continue
            out.append({**group, "metric": metric, **stat, "bootstrap_samples": samples})
    return out


def _selected_model_planned_key(key: tuple[str, ...]) -> bool:
    phase, train_config, split, _cf_type, channel_mode, fraction = key
    return (
        train_config == "default_weighted"
        and split in {"test", "test_new_agent"}
        and channel_mode == "predicted_channel"
        and abs(float(fraction) - 0.5) <= 1e-9
        and phase in {"sparse", "dense"}
    )


def _load_robustness_model_planned_rows(root: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir() or config_dir.name in {"logs"}:
            continue
        for seed_dir in sorted(config_dir.glob("seed_*")):
            model_vs_sim_path = seed_dir / "model_vs_simulator.csv"
            goal_path = seed_dir / "goal_changes.csv"
            source_path = model_vs_sim_path if model_vs_sim_path.exists() else goal_path
            if not source_path.exists():
                continue
            seed = seed_dir.name.replace("seed_", "")
            for row in read_csv_dicts(source_path):
                normal_orig = _float(row.get("normal_prob_original_goal"))
                cf_orig = _float(row.get("cf_prob_original_goal"))
                cf_top1 = _float(row.get("cf_prob_top1_goal"))
                cf_sim = _float(row.get("cf_prob_simulator_goal"))
                has_simulator_goal = _truthy(row.get("simulator_success", "1")) and row.get("simulator_cf_goal", "") != ""
                top1_agreement = _float(row.get("simulator_goal_top1_match")) if has_simulator_goal else float("nan")
                top5_agreement = _float(row.get("simulator_goal_top5_match")) if has_simulator_goal else float("nan")
                row = {
                    **row,
                    "phase": phase,
                    "train_config": config_dir.name,
                    "seed": seed,
                    "delta_original_goal": cf_orig - normal_orig,
                    "delta_selected_goal": cf_top1 - normal_orig,
                    "delta_counterfactual_margin": cf_top1 - cf_orig,
                    "delta_simulator_goal": cf_sim - normal_orig,
                    "delta_simulator_vs_selected_goal": cf_sim - cf_top1,
                    "top1_agreement": top1_agreement,
                    "top5_agreement": top5_agreement,
                    "prob_simulator_goal": cf_sim if has_simulator_goal else float("nan"),
                }
                rows.append(row)
    return rows


def _format_ci(row: dict[str, Any], percent: bool = False) -> str:
    mult = 100.0 if percent else 1.0
    mean = _float(row["mean"]) * mult
    lo = _float(row["ci_low"]) * mult
    hi = _float(row["ci_high"]) * mult
    if percent:
        return f"{mean:.1f} [{lo:.1f}, {hi:.1f}]"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def _format_p(value: Any) -> str:
    p_value = _float(value)
    if not math.isfinite(p_value):
        return "--"
    if p_value < 0.0001:
        return r"$<10^{-4}$"
    return f"{p_value:.4f}"


def _format_wide_ci(row: dict[str, Any], prefix: str, percent: bool = False) -> str:
    mult = 100.0 if percent else 1.0
    mean = _float(row[f"{prefix}_mean"]) * mult
    lo = _float(row[f"{prefix}_ci_low"]) * mult
    hi = _float(row[f"{prefix}_ci_high"]) * mult
    if percent:
        return f"{mean:.1f} [{lo:.1f}, {hi:.1f}]"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _pivot_ci(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], dict[str, dict[str, Any]]]:
    out: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in keys)
        out[key][str(row["metric"])] = row
    return out


def _write_model_planned_table(ci_rows: list[dict[str, Any]], out_dir: Path) -> None:
    keys = ["phase", "train_config", "split", "cf_type", "channel_mode", "fraction"]
    pivot = _pivot_ci(ci_rows, keys)
    selected = []
    for key, metrics in sorted(pivot.items()):
        phase, train_config, split, cf_type, channel_mode, fraction = key
        if train_config != "default_weighted" or split not in {"test", "test_new_agent"}:
            continue
        if channel_mode != "predicted_channel" or abs(float(fraction) - 0.5) > 1e-9:
            continue
        selected.append((key, metrics))

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{Model-planned trajectory counterfactual probability deltas at 50\% path reveal. Values are means with 95\% bootstrap confidence intervals clustered by seed-agent. $\Delta_{\mathrm{orig}}=p^{\mathrm{cf}}(g_{\mathrm{orig}})-p(g_{\mathrm{orig}})$; $\Delta_{\mathrm{sel}}=p^{\mathrm{cf}}(g_{\mathrm{cf}})-p(g_{\mathrm{orig}})$; $\Delta_{\mathrm{sim}}=p^{\mathrm{cf}}(g_{\mathrm{sim}})-p(g_{\mathrm{orig}})$; and $\Delta_{\mathrm{sim-cf}}=p^{\mathrm{cf}}(g_{\mathrm{sim}})-p^{\mathrm{cf}}(g_{\mathrm{cf}})$, where $g_{\mathrm{cf}}$ is the model-selected counterfactual goal and $g_{\mathrm{sim}}$ is the simulator-resampled counterfactual goal.}",
        r"  \label{tab:model_planned_delta}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llllrrrr}",
        r"    \toprule",
        r"    Regime & Split & Type & Channel & $\Delta_{\mathrm{orig}}$ & $\Delta_{\mathrm{sel}}$ & $\Delta_{\mathrm{sim}}$ & $\Delta_{\mathrm{sim-cf}}$ \\",
        r"    \midrule",
    ]
    for key, metrics in selected:
        phase, _, split, cf_type, channel_mode, _ = key
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(phase),
                    _latex_escape(split),
                    _latex_escape(cf_type),
                    _latex_escape(channel_mode.replace("_channel", "")),
                    _format_ci(metrics["delta_original_goal"]),
                    _format_ci(metrics["delta_selected_goal"]),
                    _format_ci(metrics["delta_simulator_goal"]),
                    _format_ci(metrics["delta_simulator_vs_selected_goal"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}", ""])
    write_text(out_dir / "model_planned_delta_table.tex", "\n".join(lines))


DELTA_SIGNIFICANCE_SPECS = {
    "delta_original_goal": ("less", r"$\Delta_{\mathrm{orig}}<0$"),
    "delta_selected_goal": ("greater", r"$\Delta_{\mathrm{sel}}>0$"),
    "delta_simulator_goal": ("two-sided", r"$\Delta_{\mathrm{sim}}\neq0$"),
    "delta_simulator_vs_selected_goal": ("two-sided", r"$\Delta_{\mathrm{sim-cf}}\neq0$"),
}


def _delta_significance_rows(raw_rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    group_keys = ["phase", "train_config", "split", "cf_type", "channel_mode", "fraction"]
    out: list[dict[str, Any]] = []
    for group_key, vals in sorted(_group(raw_rows, group_keys).items()):
        if not _selected_model_planned_key(group_key):
            continue
        group = dict(zip(group_keys, group_key))
        for metric, (alternative, _label) in DELTA_SIGNIFICANCE_SPECS.items():
            finite_vals = [row for row in vals if math.isfinite(_float(row.get(metric, "")))]
            stat = _random_sign_pvalue(
                finite_vals,
                metric,
                alternative,
                samples,
                seed + _stable_offset("delta-significance", metric, *group_key),
            )
            out.append(
                {
                    **group,
                    "metric": metric,
                    "alternative": alternative,
                    "n": len(finite_vals),
                    "clusters": len({_cluster_key(row) for row in finite_vals}),
                    **stat,
                }
            )
    return _holm_adjust(out)


def _write_delta_significance_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    keys = ["phase", "split", "cf_type"]
    pivot = _pivot_ci(rows, keys)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Cluster random-sign tests for the model-planned probability deltas in Table~\ref{tab:model_planned_delta}. Values are Holm-adjusted $p$-values across the displayed 50\% predicted-channel tests. One-sided alternatives are used for the original-goal decrease and model-selected-goal increase; two-sided tests are used for simulator-goal deltas.}",
        r"  \label{tab:model_planned_delta_significance}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lllrrrr}",
        r"    \toprule",
        r"    Regime & Split & Type & $p_{\mathrm{adj}}(\Delta_{\mathrm{orig}}<0)$ & $p_{\mathrm{adj}}(\Delta_{\mathrm{sel}}>0)$ & $p_{\mathrm{adj}}(\Delta_{\mathrm{sim}}\neq0)$ & $p_{\mathrm{adj}}(\Delta_{\mathrm{sim-cf}}\neq0)$ \\",
        r"    \midrule",
    ]
    for key, metrics in sorted(pivot.items()):
        phase, split, cf_type = key
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(phase),
                    _latex_escape(split),
                    _latex_escape(cf_type),
                    _format_p(metrics["delta_original_goal"]["p_adjusted"]),
                    _format_p(metrics["delta_selected_goal"]["p_adjusted"]),
                    _format_p(metrics["delta_simulator_goal"]["p_adjusted"]),
                    _format_p(metrics["delta_simulator_vs_selected_goal"]["p_adjusted"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}", ""])
    write_text(out_dir / "model_planned_delta_significance_table.tex", "\n".join(lines))


def _shuffle_agreement_pvalue(rows: list[dict[str, Any]], metric: str, samples: int, seed: int) -> dict[str, Any]:
    finite_rows = [
        row
        for row in rows
        if row.get("simulator_cf_goal", "")
        and row.get("cf_top1_goal", "")
        and row.get("cf_top5_goals", "")
        and math.isfinite(_float(row.get(metric, "")))
    ]
    if not finite_rows:
        return {"p_value": float("nan"), "observed": float("nan"), "permutation_samples": samples}

    cluster_keys = sorted({_cluster_key(row) for row in finite_rows})
    cluster_to_idx = {key: idx for idx, key in enumerate(cluster_keys)}
    cluster_indices = np.asarray([cluster_to_idx[_cluster_key(row)] for row in finite_rows], dtype=np.int64)
    cluster_counts = np.bincount(cluster_indices, minlength=len(cluster_keys)).astype(np.float64)
    observed_values = np.asarray([_float(row[metric]) for row in finite_rows], dtype=np.float64)
    observed_sums = np.bincount(cluster_indices, weights=observed_values, minlength=len(cluster_keys))
    observed = float(np.mean(observed_sums / cluster_counts))

    top5_goal_sets = [set(str(row["cf_top5_goals"]).split()) for row in finite_rows]
    goal_vocab = sorted(
        set(row["simulator_cf_goal"] for row in finite_rows)
        | set(row["cf_top1_goal"] for row in finite_rows)
        | set().union(*top5_goal_sets)
    )
    goal_to_idx = {goal: idx for idx, goal in enumerate(goal_vocab)}
    sim_goal_idx = np.asarray([goal_to_idx[row["simulator_cf_goal"]] for row in finite_rows], dtype=np.int64)
    top1_goal_idx = np.asarray([goal_to_idx[row["cf_top1_goal"]] for row in finite_rows], dtype=np.int64)
    top5_membership = np.zeros((len(finite_rows), len(goal_vocab)), dtype=bool)
    for row_idx, goal_set in enumerate(top5_goal_sets):
        for goal in goal_set:
            top5_membership[row_idx, goal_to_idx[goal]] = True
    row_indices = np.arange(len(finite_rows))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(samples):
        shuffled = rng.permutation(sim_goal_idx)
        if metric == "top1_agreement":
            perm_values = (top1_goal_idx == shuffled).astype(np.float64)
        elif metric == "top5_agreement":
            perm_values = top5_membership[row_indices, shuffled].astype(np.float64)
        else:
            raise ValueError(f"Unsupported shuffled agreement metric: {metric}")
        perm_sums = np.bincount(cluster_indices, weights=perm_values, minlength=len(cluster_keys))
        perm_stat = float(np.mean(perm_sums / cluster_counts))
        if perm_stat >= observed:
            count += 1

    return {
        "p_value": (count + 1.0) / (samples + 1.0),
        "observed": observed,
        "permutation_samples": samples,
    }


def _agreement_display_rows(
    raw_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    group_keys = ["phase", "train_config", "split", "cf_type", "channel_mode", "fraction"]
    rows: list[dict[str, Any]] = []
    shuffle_tests: list[dict[str, Any]] = []
    for group_key, vals in sorted(_group(raw_rows, group_keys).items()):
        if not _selected_model_planned_key(group_key):
            continue
        group = dict(zip(group_keys, group_key))
        top1 = _bootstrap_ci(vals, "top1_agreement", bootstrap_samples, seed + _stable_offset("agreement", "top1", *group_key))
        top5 = _bootstrap_ci(vals, "top5_agreement", bootstrap_samples, seed + _stable_offset("agreement", "top5", *group_key))
        prob = _bootstrap_ci(vals, "prob_simulator_goal", bootstrap_samples, seed + _stable_offset("agreement", "prob", *group_key))
        row_id = "|".join(group_key)
        rows.append(
            {
                **group,
                "row_id": row_id,
                "n": top1["n"],
                "clusters": top1["clusters"],
                "top1_mean": top1["mean"],
                "top1_ci_low": top1["ci_low"],
                "top1_ci_high": top1["ci_high"],
                "top5_mean": top5["mean"],
                "top5_ci_low": top5["ci_low"],
                "top5_ci_high": top5["ci_high"],
                "prob_simulator_goal_mean": prob["mean"],
                "prob_simulator_goal_ci_low": prob["ci_low"],
                "prob_simulator_goal_ci_high": prob["ci_high"],
                "bootstrap_samples": bootstrap_samples,
            }
        )
        for metric in ("top1_agreement", "top5_agreement"):
            stat = _shuffle_agreement_pvalue(
                vals,
                metric,
                permutation_samples,
                seed + _stable_offset("agreement-shuffle", metric, *group_key),
            )
            shuffle_tests.append({**group, "row_id": row_id, "metric": metric, **stat})

    adjusted_tests = _holm_adjust(shuffle_tests)
    by_row_metric = {(row["row_id"], row["metric"]): row for row in adjusted_tests}
    out = []
    for row in rows:
        top1_test = by_row_metric.get((row["row_id"], "top1_agreement"), {})
        top5_test = by_row_metric.get((row["row_id"], "top5_agreement"), {})
        out.append(
            {
                **{key: value for key, value in row.items() if key != "row_id"},
                "top1_shuffle_p": top1_test.get("p_value", float("nan")),
                "top1_shuffle_p_adjusted": top1_test.get("p_adjusted", float("nan")),
                "top5_shuffle_p": top5_test.get("p_value", float("nan")),
                "top5_shuffle_p_adjusted": top5_test.get("p_adjusted", float("nan")),
                "shuffle_permutation_samples": permutation_samples,
            }
        )
    return out


def _write_agreement_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Model-planned versus simulator-resampled counterfactual agreement at 50\% path reveal. Top-1 is the percentage of cases where the model-selected counterfactual goal equals the simulator-resampled goal; Top-5 is the percentage where the simulator goal appears in the model's edited top-5. Values are means with 95\% bootstrap confidence intervals clustered by seed-agent. $p_{\mathrm{shuffle}}$ is the Holm-adjusted within-group shuffled-goal test for Top-5 agreement above chance.}",
        r"  \label{tab:model_planned_agreement}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lllrrrrr}",
        r"    \toprule",
        r"    Regime & Split & Type & $n$ & Top-1 (\%) & Top-5 (\%) & $p^{\mathrm{cf}}(g_{\mathrm{sim}})$ & $p_{\mathrm{shuffle}}$ \\",
        r"    \midrule",
    ]
    for row in sorted(rows, key=lambda r: (str(r["phase"]), str(r["split"]), str(r["cf_type"]))):
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(str(row["phase"])),
                    _latex_escape(str(row["split"])),
                    _latex_escape(str(row["cf_type"])),
                    str(int(float(row["n"]))),
                    _format_wide_ci(row, "top1", percent=True),
                    _format_wide_ci(row, "top5", percent=True),
                    _format_wide_ci(row, "prob_simulator_goal"),
                    _format_p(row["top5_shuffle_p_adjusted"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}", ""])
    write_text(out_dir / "model_planned_agreement_table.tex", "\n".join(lines))


def _load_rollout_ci(ci_path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_dicts(ci_path):
        stage = row.get("stage", "")
        if stage == "true_counterfactual_rollouts":
            phase = "sparse"
        elif stage == "dense_true_counterfactual_rollouts":
            phase = "dense"
        else:
            continue
        metric = row.get("metric", "")
        if metric == "top1_correct":
            metric = "top1"
        elif metric == "top5_correct":
            metric = "top5"
        elif metric != "prob_counterfactual_goal":
            continue
        rows.append({**row, "phase": phase, "metric": metric})
    return rows


def _write_rollout_table(ci_rows: list[dict[str, Any]], out_dir: Path) -> None:
    keys = ["phase", "split", "cf_type", "fraction"]
    pivot = _pivot_ci(ci_rows, keys)
    selected = []
    for key, metrics in sorted(pivot.items()):
        phase, split, cf_type, fraction = key
        if split not in {"test", "test_new_agent"} or abs(float(fraction) - 0.5) > 1e-9:
            continue
        selected.append((key, metrics))

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Simulator-resampled full-trajectory counterfactual prediction at 50\% path reveal. Values are means with 95\% bootstrap confidence intervals clustered by agent. Top-1/top-5 measure agreement with the simulator-sampled counterfactual goal; $p(g_{\mathrm{sim}})$ is the mean probability assigned to that goal.}",
        r"  \label{tab:simulator_rollout}",
        r"  \begin{tabular}{llllrrr}",
        r"    \toprule",
        r"    Regime & Split & Type & $n$ & Top-1 (\%) & Top-5 (\%) & $p(g_{\mathrm{sim}})$ \\",
        r"    \midrule",
    ]
    for key, metrics in selected:
        phase, split, cf_type, _ = key
        n = int(float(metrics["top1"]["n"]))
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(phase),
                    _latex_escape(split),
                    _latex_escape(cf_type),
                    str(n),
                    _format_ci(metrics["top1"], percent=True),
                    _format_ci(metrics["top5"], percent=True),
                    _format_ci(metrics["prob_counterfactual_goal"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    write_text(out_dir / "simulator_rollout_table.tex", "\n".join(lines))


def main() -> None:
    args = parse_args()
    samples = 50 if args.smoke else args.bootstrap_samples
    permutation_samples = 100 if args.smoke else args.permutation_samples
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    robustness_roots = [
        ("sparse", resolve("outputs/diagnostics/analyses/robustness_model_planned_counterfactual_paths/sparse")),
        ("dense", resolve("outputs/diagnostics/analyses/robustness_model_planned_counterfactual_paths/dense")),
    ]
    raw_rows: list[dict[str, Any]] = []
    for phase, root in robustness_roots:
        if root.exists():
            raw_rows.extend(_load_robustness_model_planned_rows(root, phase))
    if args.smoke:
        raw_rows = raw_rows[:200]

    delta_metrics = [
        "delta_original_goal",
        "delta_selected_goal",
        "delta_counterfactual_margin",
        "delta_simulator_goal",
        "delta_simulator_vs_selected_goal",
    ]
    delta_ci = _ci_by_group(
        raw_rows,
        ["phase", "train_config", "split", "cf_type", "channel_mode", "fraction"],
        delta_metrics,
        samples,
        args.seed,
    )
    _write_csv_lf(out_dir / "model_planned_delta_summary.csv", delta_ci)
    _write_model_planned_table(delta_ci, out_dir)

    delta_significance = _delta_significance_rows(raw_rows, permutation_samples, args.seed)
    _write_csv_lf(out_dir / "model_planned_delta_significance.csv", delta_significance)
    _write_delta_significance_table(delta_significance, out_dir)

    agreement_rows = _agreement_display_rows(raw_rows, samples, permutation_samples, args.seed)
    _write_csv_lf(out_dir / "model_planned_agreement_summary.csv", agreement_rows)
    _write_agreement_table(agreement_rows, out_dir)

    rollout_ci = _load_rollout_ci(resolve("outputs/diagnostics/analyses/counterfactual_bootstrap_ci/counterfactual_ci.csv"))
    if args.smoke:
        rollout_ci = rollout_ci[:100]
    write_csv(out_dir / "simulator_rollout_summary.csv", rollout_ci)
    _write_rollout_table(rollout_ci, out_dir)

    note = [
        "# Trajectory Counterfactual Summaries",
        "",
        "Generated paper-facing summaries use probability-delta and top-k metrics only.",
        "They intentionally omit binary goal-change, node-overlap, and edge-overlap metrics from manuscript tables.",
        "",
        f"- Model-planned delta rows: {len(delta_ci)}.",
        f"- Model-planned delta significance rows: {len(delta_significance)}.",
        f"- Model-planned agreement rows: {len(agreement_rows)}.",
        f"- Simulator rollout CI rows: {len(rollout_ci)}.",
    ]
    write_text(out_dir / "summary.md", "\n".join(note) + "\n")


if __name__ == "__main__":
    main()
