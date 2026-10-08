"""
scripts/eval_goal_inference.py

Evaluate GoalPredictor, ContextGoalPredictor, and BDIBottleneck on goal
inference accuracy across path proportions for the test and test_new_agent
splits.

Produces top-1 comparison and top-1 generalization-gap plots for both:
    (test, test_new_agent) and (test_bh, test_new_agent_bh), when available.

Output
------
  results/goal_inference/results.json
    results/goal_inference/top1_comparison.png
    results/goal_inference/top1_comparison_gap.png
        results/goal_inference/top1_combined.png
    results/goal_inference/top1_comparison_bh.png
    results/goal_inference/top1_comparison_gap_bh.png
        results/goal_inference/top1_combined_bh.png

Usage
-----
  .venv/bin/python scripts/eval_goal_inference.py
    .venv/bin/python scripts/eval_goal_inference.py --bh-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from model.bdi_bottleneck import BDIBottleneckPredictor
from model.context_goal_predictor import AuxSupervisedContextGoalPredictor, ContextGoalPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.goal_predictor import GoalPredictor
from simulation.episode import EpisodeRecord

# Import eval_at_fractions from each training script
import scripts.train_goal_predictor as _gp_mod
import scripts.train_bdi_bottleneck as _bnk_mod
import scripts.train_context_goal_predictor as _ctx_mod
import scripts.train_bdi_predictor as _bdip_mod  # BDIBottleneck/Context import FRACTIONS from here

gp_eval_at_fractions  = _gp_mod.eval_at_fractions
bnk_eval_at_fractions = _bnk_mod.eval_at_fractions
ctx_eval_at_fractions = _ctx_mod.eval_at_fractions

# Fractions evaluated (includes 0.25 and 0.75 for exact table values).
FRACTIONS = _gp_mod.FRACTIONS  # original 10%-grid, used by training scripts
EVAL_FRACTIONS = sorted(set(FRACTIONS) | {0.25, 0.75})
# Fractions shown on plots (original 10%-grid only, no extra quarter points).
PLOT_FRACTIONS = FRACTIONS


def _eval_with_fractions(eval_fn, fractions, *args, **kwargs):
    """Call an eval_at_fractions function using a custom fraction list."""
    # Patch FRACTIONS in every module that defines or imports it so that the
    # underlying eval loop iterates over our extended set.
    # train_bdi_bottleneck and train_context_goal_predictor import FRACTIONS
    # from train_bdi_predictor, so train_bdi_predictor must be patched too.
    for mod in (_gp_mod, _bnk_mod, _ctx_mod, _bdip_mod):
        mod.FRACTIONS = fractions
    try:
        return eval_fn(*args, **kwargs)
    finally:
        for mod in (_gp_mod, _bnk_mod, _ctx_mod, _bdip_mod):
            mod.FRACTIONS = FRACTIONS


# ── Plot ───────────────────────────────────────────────────────────────

_COLORS = {
    "VisitFrequency":       "#9C27B0",   # purple
    "BToMInformed":         "#212121",   # black
    "GoalPredictor":        "#2196F3",   # blue
    "ContextGoalPredictor": "#FF9800",   # orange
    "AuxSupervisedContextGoalPredictor": "#E91E63",  # magenta
    "BDIBottleneck":        "#4CAF50",   # green
}
_LABELS = {
    "VisitFrequency":       "VisitFrequency (Heuristic)",
    "BToMInformed":         "BToM (Informed) [approximate upper bound]",
    "GoalPredictor":        "GoalPredictor",
    "ContextGoalPredictor": "ContextGoalPredictor",
    "AuxSupervisedContextGoalPredictor": "AuxContext (aux. sup.)",
    "BDIBottleneck":        "BDIBottleneck (Ours)",
}

_LINESTYLE = {
    "test": "solid",
    "test_new_agent": "dashed",
}


def _split_title(split_name: str) -> str:
    mapping = {
        "test": "Test",
        "test_new_agent": "Test (new agent)",
        "test_bh": "Test (BH held-out)",
        "test_new_agent_bh": "Test (new agent, BH held-out)",
    }
    return mapping.get(split_name, split_name.replace("_", " "))


def plot_top1_comparison_and_gap(
    results: dict[str, dict[str, dict[float, dict[str, float]]]],
    split_test: str,
    split_new: str,
    out_path_comparison: Path,
    out_path_gap: Path,
    title_suffix: str = "",
    paper: bool = False,
) -> bool:
    """
    Produce two separate top-1 figures for a split pair:
      1) comparison curves for split_test vs split_new
      2) generalization gap curve = split_test - split_new
    Returns True if both splits were present for at least one model.
    """
    pct = [int(f * 100) for f in PLOT_FRACTIONS]
    has_pair_data = False

    # ── Comparison plot ─────────────────────────────────────────────
    fig_acc, ax_acc = plt.subplots(1, 1, figsize=(7.5, 4.8))

    for model_name, split_dict in results.items():
        if split_test not in split_dict or split_new not in split_dict:
            continue
        has_pair_data = True

        ys_test  = [split_dict[split_test][f]["top1"] * 100 for f in PLOT_FRACTIONS]
        ys_new   = [split_dict[split_new][f]["top1"] * 100 for f in PLOT_FRACTIONS]
        ys_gap   = [t - n for t, n in zip(ys_test, ys_new)]
        std_test = [split_dict[split_test][f].get("top1_std", 0.0) * 100 for f in PLOT_FRACTIONS]
        std_new  = [split_dict[split_new][f].get("top1_std", 0.0) * 100 for f in PLOT_FRACTIONS]
        color    = _COLORS.get(model_name, "gray")

        ax_acc.plot(
            pct, ys_test,
            color=color,
            linestyle="solid",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=f"{_LABELS.get(model_name, model_name)} — {_split_title(split_test)}",
        )
        if any(s > 0 for s in std_test):
            ax_acc.fill_between(pct, [y-s for y,s in zip(ys_test,std_test)],
                                [y+s for y,s in zip(ys_test,std_test)], alpha=0.15, color=color)
        ax_acc.plot(
            pct, ys_new,
            color=color,
            linestyle="dashed",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=f"{_LABELS.get(model_name, model_name)} — {_split_title(split_new)}",
        )
        if any(s > 0 for s in std_new):
            ax_acc.fill_between(pct, [y-s for y,s in zip(ys_new,std_new)],
                                [y+s for y,s in zip(ys_new,std_new)], alpha=0.15, color=color)

    ax_acc.set_title(
        f"Top-1 Accuracy{title_suffix}",
        fontsize=12,
        fontweight="bold" if not paper else None,
    )
    ax_acc.set_xlabel("Path revealed (%)", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xlim(5, 105)
    ax_acc.set_ylim(0, 105)
    ax_acc.set_xticks(pct)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    ax_acc.legend(fontsize=8.8, frameon=False, ncol=2)

    dpi = 300 if paper else 150
    fig_acc.tight_layout(pad=1.0)
    fig_acc.savefig(out_path_comparison, dpi=dpi, bbox_inches="tight")
    plt.close(fig_acc)
    print(f"  Saved → {out_path_comparison}")

    # ── Gap plot ────────────────────────────────────────────────────
    fig_gap, ax_gap = plt.subplots(1, 1, figsize=(7.0, 4.8))

    for model_name, split_dict in results.items():
        if split_test not in split_dict or split_new not in split_dict:
            continue

        ys_test  = [split_dict[split_test][f]["top1"] * 100 for f in PLOT_FRACTIONS]
        ys_new   = [split_dict[split_new][f]["top1"] * 100 for f in PLOT_FRACTIONS]
        ys_gap   = [t - n for t, n in zip(ys_test, ys_new)]
        std_test = [split_dict[split_test][f].get("top1_std", 0.0) * 100 for f in PLOT_FRACTIONS]
        std_new  = [split_dict[split_new][f].get("top1_std", 0.0) * 100 for f in PLOT_FRACTIONS]
        std_gap  = [np.sqrt(a**2 + b**2) for a, b in zip(std_test, std_new)]
        color    = _COLORS.get(model_name, "gray")

        ax_gap.plot(
            pct, ys_gap,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=_LABELS.get(model_name, model_name),
        )
        if any(s > 0 for s in std_gap):
            ax_gap.fill_between(pct, [g-s for g,s in zip(ys_gap,std_gap)],
                                [g+s for g,s in zip(ys_gap,std_gap)], alpha=0.15, color=color)

    ax_gap.set_title(
        f"Generalization Gap ({_split_title(split_test)} - {_split_title(split_new)}){title_suffix}",
        fontsize=12,
        fontweight="bold" if not paper else None,
    )
    ax_gap.set_xlabel("Path revealed (%)", fontsize=11)
    ax_gap.set_ylabel("Gap (percentage points)", fontsize=11)
    ax_gap.set_xlim(5, 105)
    ax_gap.set_xticks(pct)
    ax_gap.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.spines["top"].set_visible(False)
    ax_gap.spines["right"].set_visible(False)
    ax_gap.legend(fontsize=9.5, frameon=False)

    fig_gap.tight_layout(pad=1.0)
    fig_gap.savefig(out_path_gap, dpi=dpi, bbox_inches="tight")
    plt.close(fig_gap)
    print(f"  Saved → {out_path_gap}")

    return has_pair_data


def plot_top1_combined(
    results: dict[str, dict[str, dict[float, dict[str, float]]]],
    split_test: str,
    split_new: str,
    out_path: Path,
    title_suffix: str = "",
    paper: bool = False,
) -> bool:
    """
    Produce one side-by-side figure with:
      Left: top-1 curves for split_test vs split_new
      Right: generalization gap curve = split_test - split_new
    Uses a single shared legend for model colors and split line styles.
    """
    pct = [int(f * 100) for f in PLOT_FRACTIONS]
    has_pair_data = False

    fig, (ax_acc, ax_gap) = plt.subplots(
        1, 2,
        figsize=(11.0, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
        sharex=True,
    )

    for model_name, split_dict in results.items():
        if split_test not in split_dict or split_new not in split_dict:
            continue
        has_pair_data = True

        ys_test  = [split_dict[split_test][f]["top1"] * 100 for f in PLOT_FRACTIONS]
        ys_new   = [split_dict[split_new][f]["top1"] * 100 for f in PLOT_FRACTIONS]
        ys_gap   = [t - n for t, n in zip(ys_test, ys_new)]
        std_test = [split_dict[split_test][f].get("top1_std", 0.0) * 100 for f in PLOT_FRACTIONS]
        std_new  = [split_dict[split_new][f].get("top1_std", 0.0) * 100 for f in PLOT_FRACTIONS]
        std_gap  = [np.sqrt(a**2 + b**2) for a, b in zip(std_test, std_new)]
        color    = _COLORS.get(model_name, "gray")

        ax_acc.plot(
            pct, ys_test,
            color=color,
            linestyle="solid",
            linewidth=2.5,
            marker="o",
            markersize=5,
        )
        if any(s > 0 for s in std_test):
            ax_acc.fill_between(pct, [y-s for y,s in zip(ys_test,std_test)],
                                [y+s for y,s in zip(ys_test,std_test)], alpha=0.15, color=color)

        ax_gap.plot(
            pct, ys_gap,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=5,
        )
        if any(s > 0 for s in std_gap):
            ax_gap.fill_between(pct, [g-s for g,s in zip(ys_gap,std_gap)],
                                [g+s for g,s in zip(ys_gap,std_gap)], alpha=0.15, color=color)

    ax_acc.set_title(
        f"Top-1 Accuracy — Test{title_suffix}",
        fontsize=12,
        fontweight="bold" if not paper else None,
    )
    ax_acc.set_xlabel("Path revealed (%)", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xlim(5, 105)
    ax_acc.set_ylim(0, 105)
    ax_acc.set_xticks([int(f * 100) for f in PLOT_FRACTIONS])
    ax_acc.grid(True, alpha=0.3)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    ax_gap.set_title(
        f"Generalization Gap ({_split_title(split_test)} - {_split_title(split_new)}){title_suffix}",
        fontsize=12,
        fontweight="bold" if not paper else None,
    )
    ax_gap.set_xlabel("Path revealed (%)", fontsize=11)
    ax_gap.set_ylabel("Gap (percentage points)", fontsize=11)
    ax_gap.set_xlim(5, 105)
    ax_gap.set_xticks([int(f * 100) for f in PLOT_FRACTIONS])
    ax_gap.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.spines["top"].set_visible(False)
    ax_gap.spines["right"].set_visible(False)

    model_handles = [
        Line2D(
            [0], [0],
            color=_COLORS.get(model_name, "gray"),
            linestyle="solid",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=_LABELS.get(model_name, model_name),
        )
        for model_name in results.keys()
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9.0,
        bbox_to_anchor=(0.5, 1.05),
    )

    dpi = 300 if paper else 150
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93], pad=1.0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

    return has_pair_data


def plot_top5_split(
    results: dict[str, dict[str, dict[float, dict[str, float]]]],
    split_name: str,
    out_path: Path,
    paper: bool = False,
) -> bool:
    """Plot top-5 accuracy curves for all models on a single split."""
    pct = [int(f * 100) for f in PLOT_FRACTIONS]
    has_data = False

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))

    for model_name, split_dict in results.items():
        if split_name not in split_dict:
            continue
        has_data = True
        ys = [split_dict[split_name][f]["top5"] * 100 for f in PLOT_FRACTIONS]
        ax.plot(
            pct, ys,
            color=_COLORS.get(model_name, "gray"),
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=_LABELS.get(model_name, model_name),
        )

    ax.set_title(
        f"Top-5 Accuracy — {_split_title(split_name)}",
        fontsize=12,
        fontweight="bold" if not paper else None,
    )
    ax.set_xlabel("Path revealed (%)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlim(5, 105)
    ax.set_ylim(0, 105)
    ax.set_xticks(pct)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9.5, frameon=False)

    dpi = 300 if paper else 150
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return has_data


# ── Heuristic baselines ────────────────────────────────────────────────

def nearest_poi_eval_at_fractions(
    records:        list[EpisodeRecord],
    node_coords:    dict[int, tuple[float, float]],
    poi_to_idx:     dict[str, int],
    poi_coords_arr: np.ndarray,
    fractions:      list[float],
) -> dict[float, dict[str, float]]:
    """
    Nearest-POI heuristic baseline.

    At each path fraction, predicts the POI geographically closest to the
    agent's last observed node position using Euclidean lat/lon distance.
    No learning, no context — purely spatial.
    """
    results: dict[float, dict[str, float]] = {}
    for frac in fractions:
        correct1 = 0
        correct5 = 0
        for r in records:
            T = len(r.path)
            k = max(1, round(frac * T))
            cur_node = r.path[k - 1]
            lat, lon = node_coords.get(cur_node, (0.0, 0.0))
            pos = np.array([lat, lon], dtype=np.float64)
            dists = np.linalg.norm(poi_coords_arr - pos, axis=1)
            top5_idx = np.argsort(dists)[:5]
            true_idx = poi_to_idx[r.goal_poi]
            if top5_idx[0] == true_idx:
                correct1 += 1
            if true_idx in top5_idx:
                correct5 += 1
        n = len(records)
        results[frac] = {"top1": correct1 / n, "top5": correct5 / n}
    return results


def visit_freq_eval_at_fractions(
    records:     list[EpisodeRecord],
    poi_to_idx:  dict[str, int],
    poi_order:   list[str],
    poi_to_cat:  dict[str, str],
    fractions:   list[float],
    K:           int = 10,
    rng_seed:    int = 42,
) -> dict[float, dict[str, float]]:
    """
    Agent visit-frequency baseline.

    Uses the same K desire-context episodes the neural models receive (K
    randomly sampled same-agent past trajectories) and predicts the POI that
    appeared most often as a goal in that context. Ties are broken first by
    total category frequency, then by POI index (lower wins).

    Path-fraction invariant: the desire context is the same regardless of how
    much of the current trajectory is revealed, so accuracy is identical at
    every fraction. The constant value is returned for all requested fractions.
    """
    from collections import Counter, defaultdict

    rng = np.random.default_rng(rng_seed)
    num_pois = len(poi_order)

    # agent_id -> ordered list of flat record indices (simulation order)
    agent_idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        agent_idx[r.agent_id].append(i)

    correct1 = 0
    correct5 = 0

    for flat_idx, r in enumerate(records):
        agent_idxs = agent_idx[r.agent_id]
        others = [i for i in agent_idxs if i != flat_idx]

        # Mirror the desire-context sampling from train_context_goal_predictor
        if len(others) == 0:
            d_src = [flat_idx] * K
        elif len(others) < K:
            d_src = rng.choice(others, size=K, replace=True).tolist()
        else:
            d_src = rng.choice(others, size=K, replace=False).tolist()

        poi_counts: Counter = Counter(records[i].goal_poi for i in d_src)

        # Category-level counts for tie-breaking
        cat_counts: dict[str, int] = defaultdict(int)
        for poi_id, cnt in poi_counts.items():
            cat_counts[poi_to_cat[poi_id]] += cnt

        # Rank all POIs: poi_count DESC, cat_count DESC, poi_idx ASC
        ranked = sorted(
            range(num_pois),
            key=lambda idx: (
                -poi_counts.get(poi_order[idx], 0),
                -cat_counts.get(poi_to_cat[poi_order[idx]], 0),
                idx,
            ),
        )
        top5 = ranked[:5]
        true_idx = poi_to_idx[r.goal_poi]
        if top5[0] == true_idx:
            correct1 += 1
        if true_idx in top5:
            correct5 += 1

    n = len(records)
    frac_result = {"top1": correct1 / n, "top5": correct5 / n}
    return {frac: frac_result for frac in fractions}


# ── BToM (Bayesian Theory of Mind) ────────────────────────────────────────

_BTOM_INF = 1e6   # sentinel distance for unreachable nodes
_BTOM_BETA_GRID = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


def _precompute_btom_distances(G, poi_node_ids: list[int], cache_path: Path) -> list[dict]:
    """
    For each POI, run single-source Dijkstra from that POI's node over the
    undirected walking graph. Returns dist_tables[poi_idx][node_id] = metres.
    Cached to cache_path as a pickle so the expensive step runs only once.
    """
    import pickle
    import networkx as nx

    if cache_path.exists():
        print(f"  Loading BToM distances from cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print("  Precomputing BToM distances (Dijkstra from each POI) …")
    import networkx as nx
    G_ud = G.to_undirected()
    dist_tables = []
    for poi_node in poi_node_ids:
        lengths = nx.single_source_dijkstra_path_length(G_ud, poi_node, weight="length")
        dist_tables.append(dict(lengths))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(dist_tables, fh)
    print(f"  Cached → {cache_path}")
    return dist_tables


def _build_neighbor_table(G) -> dict[int, list[int]]:
    """Build {node_id: [neighbor_ids]} from the undirected walking graph."""
    G_ud = G.to_undirected()
    return {int(u): [int(v) for v in G_ud.neighbors(u)] for u in G_ud.nodes()}


def btom_uniform_eval_at_fractions(
    records:        list[EpisodeRecord],
    dist_tables:    list[dict],
    neighbor_table: dict[int, list[int]],
    poi_to_idx:     dict[str, int],
    fractions:      list[float],
    beta:           float,
) -> dict[float, dict[str, float]]:
    """
    BToM baseline with uniform prior.

    For each path prefix of length k, accumulates the Boltzmann log-likelihood
    of each observed step under every goal hypothesis:
        log P(v | u, g) = log softmax(-beta * dist[g][nbr] for nbr in N(u)) at v

    Since the prior P(goal=g) is uniform it cancels in argmax and is omitted.
    Cumulative log-likelihoods are built once per record and sampled at each
    fraction, avoiding redundant computation across fractions.
    """
    num_pois = len(dist_tables)
    per_frac_c1 = {f: 0 for f in fractions}
    per_frac_c5 = {f: 0 for f in fractions}

    for r in records:
        T = len(r.path)
        true_idx = poi_to_idx[r.goal_poi]

        # Build cumulative log-lik vectors indexed by prefix length k.
        # k=1 → 0 steps observed → uniform log-lik (zeros).
        cum_ll = np.zeros(num_pois)
        k_to_ll: dict[int, np.ndarray] = {1: cum_ll.copy()}

        for t in range(T - 1):
            u, v = r.path[t], r.path[t + 1]
            nbrs = neighbor_table.get(u, [])
            if nbrs:
                # dist_matrix: (num_pois, n_nbrs)
                dist_matrix = np.array(
                    [[dist_tables[g].get(nbr, _BTOM_INF) for nbr in nbrs]
                     for g in range(num_pois)],
                    dtype=np.float64,
                )
                log_scores = -beta * dist_matrix
                row_max = log_scores.max(axis=1, keepdims=True)
                log_softmax = (log_scores - row_max
                               - np.log(np.exp(log_scores - row_max).sum(axis=1, keepdims=True)))
                try:
                    v_col = nbrs.index(v)
                    cum_ll += log_softmax[:, v_col]
                except ValueError:
                    pass   # v not a graph-neighbor of u; skip step

            k_to_ll[t + 2] = cum_ll.copy()

        for frac in fractions:
            k = max(1, round(frac * T))
            ll = k_to_ll.get(k, k_to_ll[max(kk for kk in k_to_ll if kk <= k)])
            top5 = np.argsort(-ll)[:5]
            if top5[0] == true_idx:
                per_frac_c1[frac] += 1
            if true_idx in top5:
                per_frac_c5[frac] += 1

    n = len(records)
    return {f: {"top1": per_frac_c1[f] / n, "top5": per_frac_c5[f] / n} for f in fractions}


def _tune_btom_beta(
    records:        list[EpisodeRecord],
    dist_tables:    list[dict],
    neighbor_table: dict[int, list[int]],
    poi_to_idx:     dict[str, int],
    fractions:      list[float],
    beta_grid:      list[float] = _BTOM_BETA_GRID,
) -> float:
    """Grid-search beta on the provided records; returns the value maximising mean top-1."""
    best_beta, best_score = beta_grid[0], -1.0
    for beta in beta_grid:
        res = btom_uniform_eval_at_fractions(
            records, dist_tables, neighbor_table, poi_to_idx, fractions, beta
        )
        score = np.mean([v["top1"] for v in res.values()])
        print(f"    beta={beta:.4f}  mean-top1={score*100:.2f}%")
        if score > best_score:
            best_score, best_beta = score, beta
    return best_beta


def btom_informed_eval_at_fractions(
    records:        list[EpisodeRecord],
    dist_tables:    list[dict],
    neighbor_table: dict[int, list[int]],
    poi_to_idx:     dict[str, int],
    poi_order:      list[str],
    fractions:      list[float],
    beta:           float,
    K:              int = 10,
    rng_seed:       int = 42,
) -> dict[float, dict[str, float]]:
    """
    BToM baseline with visit-frequency informed prior (Laplace smoothing).

    Prior: from K desire-context trajectories (same sampling as VisitFrequency),
    compute visit counts per POI and apply add-1 Laplace smoothing:
        log_prior[g] = log(counts[g] + 1) - log(sum(counts + 1))

    Posterior: log_prior + cumulative Boltzmann log-likelihood (as in BToMUniform).
    """
    from collections import defaultdict

    rng = np.random.default_rng(rng_seed)
    num_pois = len(dist_tables)

    agent_idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        agent_idx[r.agent_id].append(i)

    per_frac_c1 = {f: 0 for f in fractions}
    per_frac_c5 = {f: 0 for f in fractions}

    for flat_idx, r in enumerate(records):
        agent_idxs = agent_idx[r.agent_id]
        others = [i for i in agent_idxs if i != flat_idx]

        if len(others) == 0:
            d_src = [flat_idx] * K
        elif len(others) < K:
            d_src = rng.choice(others, size=K, replace=True).tolist()
        else:
            d_src = rng.choice(others, size=K, replace=False).tolist()

        # Build Laplace-smoothed log-prior from desire-context visit frequencies
        counts = np.ones(num_pois, dtype=np.float64)  # add-1 Laplace smoothing
        for ctx_idx in d_src:
            ctx_poi = records[ctx_idx].goal_poi
            if ctx_poi in poi_to_idx:
                counts[poi_to_idx[ctx_poi]] += 1.0
        log_prior = np.log(counts) - np.log(counts.sum())

        # Cumulative Boltzmann log-likelihood (same as BToMUniform)
        T = len(r.path)
        true_idx = poi_to_idx[r.goal_poi]

        cum_ll = np.zeros(num_pois)
        k_to_ll: dict[int, np.ndarray] = {1: cum_ll.copy()}

        for t in range(T - 1):
            u, v = r.path[t], r.path[t + 1]
            nbrs = neighbor_table.get(u, [])
            if nbrs:
                dist_matrix = np.array(
                    [[dist_tables[g].get(nbr, _BTOM_INF) for nbr in nbrs]
                     for g in range(num_pois)],
                    dtype=np.float64,
                )
                log_scores = -beta * dist_matrix
                row_max = log_scores.max(axis=1, keepdims=True)
                log_softmax = (log_scores - row_max
                               - np.log(np.exp(log_scores - row_max).sum(axis=1, keepdims=True)))
                try:
                    v_col = nbrs.index(v)
                    cum_ll += log_softmax[:, v_col]
                except ValueError:
                    pass

            k_to_ll[t + 2] = cum_ll.copy()

        for frac in fractions:
            k = max(1, round(frac * T))
            ll = k_to_ll.get(k, k_to_ll[max(kk for kk in k_to_ll if kk <= k)])
            posterior = log_prior + ll
            top5 = np.argsort(-posterior)[:5]
            if top5[0] == true_idx:
                per_frac_c1[frac] += 1
            if true_idx in top5:
                per_frac_c5[frac] += 1

    n = len(records)
    return {f: {"top1": per_frac_c1[f] / n, "top5": per_frac_c5[f] / n} for f in fractions}


def _tune_btom_beta_informed(
    records:        list[EpisodeRecord],
    dist_tables:    list[dict],
    neighbor_table: dict[int, list[int]],
    poi_to_idx:     dict[str, int],
    poi_order:      list[str],
    fractions:      list[float],
    beta_grid:      list[float] = _BTOM_BETA_GRID,
    K:              int = 10,
) -> float:
    """Grid-search beta for BToMInformed; returns value maximising mean top-1."""
    best_beta, best_score = beta_grid[0], -1.0
    for beta in beta_grid:
        res = btom_informed_eval_at_fractions(
            records, dist_tables, neighbor_table, poi_to_idx, poi_order,
            fractions, beta, K=K,
        )
        score = np.mean([v["top1"] for v in res.values()])
        print(f"    beta={beta:.4f}  mean-top1={score*100:.2f}%")
        if score > best_score:
            best_score, best_beta = score, beta
    return best_beta


# ── Multi-seed aggregation ────────────────────────────────────────────


def _aggregate_seeds(
    per_seed: list[dict[str, dict[float, dict[str, float]]]]
) -> dict[str, dict[float, dict[str, float]]]:
    """Average top1/top5 across seeds; add _std entries. n=1 gives std=0 (no band)."""
    if not per_seed:
        return {}
    agg: dict[str, dict[float, dict[str, float]]] = {}
    for split in per_seed[0]:
        agg[split] = {}
        for frac in per_seed[0][split]:
            top1s = [s[split][frac]["top1"] for s in per_seed]
            top5s = [s[split][frac]["top5"] for s in per_seed]
            agg[split][frac] = {
                "top1":     float(np.mean(top1s)),
                "top5":     float(np.mean(top5s)),
                "top1_std": float(np.std(top1s, ddof=0)),
                "top5_std": float(np.std(top5s, ddof=0)),
            }
    return agg


# ── LaTeX table generation ─────────────────────────────────────────────

_TABLE_MODEL_LABELS = {
    "VisitFrequency":       "VisitFrequency",
    "BToMInformed":         "BToM (Informed)",
    "GoalPredictor":        "GoalPredictor",
    "ContextGoalPredictor": "ContextGoalPredictor",
    "AuxSupervisedContextGoalPredictor": r"\shortstack[l]{AuxContext\\(aux. sup.)}",
    "BDIBottleneck":        "BDIBottleneck (Ours)",
}
_TABLE_MODEL_ORDER = list(_TABLE_MODEL_LABELS.keys())
_CEILING_LABEL = "BToM (Informed)"  # underlined in tables; col-max computed over non-ceiling rows only

_BH_REPORT_FRACS = [0.25, 0.50, 0.75, 1.00]


def _generate_bh_table_tex(
    results: dict,
    out_path: Path,
) -> None:
    """
    Write the belief-heldout top-1 table as a self-contained LaTeX file.

    Reads test_bh and test_new_agent_bh results at 25/50/75/100% and
    produces a tabular with:
      - BToM (Informed) row underlined (approximate upper bound / ceiling)
      - Bold marks the best result per column among non-ceiling models only
    """
    bh_splits = ["test_bh", "test_new_agent_bh"]

    # Collect models that have BH data in the expected order
    available = [
        (key, _TABLE_MODEL_LABELS[key])
        for key in _TABLE_MODEL_ORDER
        if key in results and all(s in results[key] for s in bh_splits)
    ]
    if not available:
        return

    # Build value matrix: rows=models, cols=(test_bh × 4 fracs, tna_bh × 4 fracs)
    num_cols = len(_BH_REPORT_FRACS) * 2
    rows_data: list[tuple[str, list[float]]] = []

    for key, label in available:
        row: list[float] = []
        for split in bh_splits:
            for frac in _BH_REPORT_FRACS:
                v = results[key][split].get(frac, {}).get("top1", 0.0) * 100
                row.append(v)
        rows_data.append((label, row))

    # Column max computed only over non-ceiling rows
    non_ceiling_rows = [(lbl, row) for lbl, row in rows_data if lbl != _CEILING_LABEL]
    non_ceil_col_max = [
        max((row[i] for _, row in non_ceiling_rows), default=0.0)
        for i in range(num_cols)
    ]

    def fmt(v: float, label: str, col_i: int) -> str:
        s = f"{v:.2f}"
        if label == _CEILING_LABEL:
            return r"\underline{" + s + "}"
        if abs(v - non_ceil_col_max[col_i]) < 1e-6:
            s = r"\textbf{" + s + "}"
        return s

    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.5pt}",
        (r"  \caption{Top-1 accuracy (\%) at 25/50/75/100\% path reveal for belief heldout splits."
         r" AuxContext is the unbottlenecked context model trained with the same auxiliary desire/belief losses as BDIBottleneck."
         r" \underline{Underline}: BToM (Informed), approximate upper bound on achievable accuracy."
         r" \textbf{Bold}: best result per column among learned models.}"),
        r"  \label{tab:goal_inference_top1_quarters_bh}",
        r"  \begin{tabular}{lrrrrrrrr}",
        r"    \toprule",
        (r"    & \multicolumn{4}{c}{Test (belief heldout)}"
         r" & \multicolumn{4}{c}{Test (new agent, belief heldout)} \\"),
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"    Model & 25\% & 50\% & 75\% & 100\% & 25\% & 50\% & 75\% & 100\% \\",
        r"    \midrule",
    ]
    for label, row in rows_data:
        display_label = (
            r"\textit{BToM (Informed) [approximate upper bound]}"
            if label == _CEILING_LABEL else label
        )
        cells = " & ".join(fmt(row[i], label, i) for i in range(num_cols))
        lines.append(f"    {display_label} & {cells} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  Saved → {out_path}")


# ── Helpers ────────────────────────────────────────────────────────────

def _load_records(path: Path) -> list[EpisodeRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_json(line))
    return records


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare GoalPredictor and BDIPredictor on goal inference."
    )
    p.add_argument("--goal-ckpt",  nargs="+", default=["outputs/goal_predictor/best.pt"],
                   metavar="PT", help="GoalPredictor checkpoint(s), one per seed")
    p.add_argument("--ctx-ckpt",   nargs="+", default=["outputs/context_goal/best.pt"],
                   metavar="PT", help="ContextGoalPredictor checkpoint(s), one per seed")
    p.add_argument("--bnk-ckpt",   nargs="+", default=["outputs/bdi_bottleneck/best.pt"],
                   metavar="PT", help="BDIBottleneck checkpoint(s), one per seed")
    p.add_argument("--splits-dir", default="data/splits",                    metavar="DIR")
    p.add_argument("--splits",     nargs="+",
                   default=["test", "test_new_agent", "test_bh", "test_new_agent_bh"])
    p.add_argument("--bh-only",    action="store_true",
                   help="Evaluate only held-out BH splits: test_bh and test_new_agent_bh")
    p.add_argument("--out-subdir", default="goal_inference", metavar="DIR",
                   help="Results subdirectory under results/ (default: goal_inference)")
    p.add_argument("--K",          type=int, default=10, metavar="K")
    p.add_argument("--K-recent",   type=int, default=5,  metavar="K", dest="K_recent")
    p.add_argument("--batch",      type=int, default=64, metavar="N")
    p.add_argument("--no-cuda",    action="store_true")
    p.add_argument("--paper",      action="store_true",
                   help="Paper-ready output (300 DPI)")
    return p.parse_args()


def main() -> None:
    args       = parse_args()

    if args.bh_only:
        args.splits = ["test_bh", "test_new_agent_bh"]

    splits_dir = ROOT / args.splits_dir
    out_dir    = ROOT / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────
    if args.no_cuda:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── POI config ────────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order  = [p["id"] for cd in poi_cfg["categories"].values() for p in cd["pois"]]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    # ── Graph ─────────────────────────────────────────────────────────
    import osmnx as ox
    print("Loading graph …")
    G           = ox.load_graphml(ROOT / "data" / "ucsd_walk.graphml")
    node_coords = node_coords_from_graph(G)
    print(f"  {len(node_coords):,} nodes\n")

    # ── POI node ID lookup (needed by BToM) ──────────────────────────────
    poi_node_id_by_id = {
        p["id"]: p["node_id"]
        for cat_data in poi_cfg["categories"].values()
        for p in cat_data["pois"]
    }
    poi_node_id_list = [poi_node_id_by_id[pid] for pid in poi_order]

    # ── POI-to-category map for VisitFrequency baseline ───────────────
    poi_to_cat = {
        p["id"]: cat_name
        for cat_name, cat_data in poi_cfg["categories"].items()
        for p in cat_data["pois"]
    }

    # ── BToM precomputation and beta tuning ───────────────────────────
    print("Setting up BToM baseline …")
    dist_tables    = _precompute_btom_distances(G, poi_node_id_list,
                                                ROOT / "data" / "btom_distances.pkl")
    neighbor_table = _build_neighbor_table(G)

    val_path = splits_dir / "val.jsonl"
    if val_path.exists():
        val_records = _load_records(val_path)
        print(f"  Tuning BToMInformed beta on val ({len(val_records)} episodes) …")
        btom_informed_beta = _tune_btom_beta_informed(
            val_records, dist_tables, neighbor_table, poi_to_idx, poi_order,
            EVAL_FRACTIONS, K=args.K,
        )
    else:
        btom_informed_beta = 0.005
        print(f"  val.jsonl not found — using default beta={btom_informed_beta}")
    print(f"  Using BToMInformed beta = {btom_informed_beta}\n")

    # ── Load records once (shared across seeds) ───────────────────────
    all_records: dict[str, list] = {}
    for split_name in args.splits:
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping.")
            continue
        all_records[split_name] = _load_records(path)
        print(f"  Loaded {len(all_records[split_name]):>5} records  ← {split_name}")
    print()

    # ── results[model][split][fraction] = {top1, top5[, top1_std, top5_std]} ──
    results: dict[str, dict[str, dict]] = {
        "VisitFrequency":       {},
        "BToMInformed":         {},
        "GoalPredictor":        {},
        "ContextGoalPredictor": {},
        "BDIBottleneck":        {},
    }

    # Non-seeded baselines (deterministic — evaluate once)
    for split_name, records in all_records.items():
        print(f"  {split_name}  ({len(records)} episodes)")
        print(f"    Running VisitFrequency …")
        results["VisitFrequency"][split_name] = visit_freq_eval_at_fractions(
            records, poi_to_idx, poi_order, poi_to_cat, EVAL_FRACTIONS, K=args.K,
        )
        print(f"    Running BToM (Informed Prior) …")
        results["BToMInformed"][split_name] = btom_informed_eval_at_fractions(
            records, dist_tables, neighbor_table, poi_to_idx, poi_order,
            EVAL_FRACTIONS, btom_informed_beta, K=args.K,
        )
    print()

    # Multi-seed learned models
    gp_per_seed:  list[dict[str, dict]] = []
    ctx_per_seed: list[dict[str, dict]] = []
    bnk_per_seed: list[dict[str, dict]] = []

    n_seeds = max(len(args.goal_ckpt), len(args.ctx_ckpt), len(args.bnk_ckpt))
    for seed_i in range(n_seeds):
        # ── GoalPredictor ─────────────────────────────────────────────
        if seed_i < len(args.goal_ckpt):
            gp_path = ROOT / args.goal_ckpt[seed_i]
            print(f"Loading GoalPredictor (seed {seed_i}) from {gp_path}")
            gp_ckpt   = torch.load(gp_path, map_location=device, weights_only=False)
            gp_state  = gp_ckpt["model_state"]
            d_model_gp = gp_state["coord_proj.weight"].shape[0]
            num_pois   = gp_state["head.weight"].shape[0]
            goal_model = GoalPredictor(num_pois=num_pois, d_model=d_model_gp).to(device)
            goal_model.load_state_dict(gp_state)
            goal_model.eval()
            seed_gp: dict[str, dict] = {}
            for split_name, records in all_records.items():
                print(f"    GP [{seed_i}] → {split_name} …")
                seed_gp[split_name] = _eval_with_fractions(
                    gp_eval_at_fractions, EVAL_FRACTIONS,
                    goal_model, records, node_coords, poi_to_idx, device,
                    batch_size=args.batch,
                )
            gp_per_seed.append(seed_gp)
            del goal_model

        # ── ContextGoalPredictor ──────────────────────────────────────
        if seed_i < len(args.ctx_ckpt):
            ctx_path = ROOT / args.ctx_ckpt[seed_i]
            print(f"Loading ContextGoalPredictor (seed {seed_i}) from {ctx_path}")
            ctx_ckpt_  = torch.load(ctx_path, map_location=device, weights_only=False)
            ctx_state  = ctx_ckpt_["model_state"]
            d_model_ctx = ctx_state["coord_proj.weight"].shape[0]
            ctx_model  = ContextGoalPredictor(num_pois=num_pois, d_model=d_model_ctx).to(device)
            ctx_model.load_state_dict(ctx_state)
            ctx_model.eval()
            seed_ctx: dict[str, dict] = {}
            for split_name, records in all_records.items():
                print(f"    CTX [{seed_i}] → {split_name} …")
                seed_ctx[split_name] = _eval_with_fractions(
                    ctx_eval_at_fractions, EVAL_FRACTIONS,
                    ctx_model, records, node_coords, poi_to_idx, device,
                    K=args.K, K_recent=args.K_recent, batch_size=args.batch,
                )
            ctx_per_seed.append(seed_ctx)
            del ctx_model

        # ── BDIBottleneck ─────────────────────────────────────────────
        if seed_i < len(args.bnk_ckpt):
            bnk_path = ROOT / args.bnk_ckpt[seed_i]
            print(f"Loading BDIBottleneck (seed {seed_i}) from {bnk_path}")
            bnk_ckpt_  = torch.load(bnk_path, map_location=device, weights_only=False)
            bnk_state  = bnk_ckpt_["model_state"]
            d_model_bn = bnk_state["coord_proj.weight"].shape[0]
            bnk_model  = BDIBottleneckPredictor(num_pois=num_pois, d_model=d_model_bn).to(device)
            bnk_model.load_state_dict(bnk_state)
            bnk_model.eval()
            seed_bnk: dict[str, dict] = {}
            for split_name, records in all_records.items():
                print(f"    BNK [{seed_i}] → {split_name} …")
                seed_bnk[split_name] = _eval_with_fractions(
                    bnk_eval_at_fractions, EVAL_FRACTIONS,
                    bnk_model, records, node_coords, poi_to_idx, device,
                    K=args.K, K_recent=args.K_recent, batch_size=args.batch,
                )
            bnk_per_seed.append(seed_bnk)
            del bnk_model

    # Aggregate across seeds
    if gp_per_seed:
        print(f"\nAggregating GoalPredictor across {len(gp_per_seed)} seed(s)")
        results["GoalPredictor"] = _aggregate_seeds(gp_per_seed)
    if ctx_per_seed:
        print(f"Aggregating ContextGoalPredictor across {len(ctx_per_seed)} seed(s)")
        results["ContextGoalPredictor"] = _aggregate_seeds(ctx_per_seed)
    if bnk_per_seed:
        print(f"Aggregating BDIBottleneck across {len(bnk_per_seed)} seed(s)")
        results["BDIBottleneck"] = _aggregate_seeds(bnk_per_seed)

    # Print summary table (means only)
    for split_name in all_records:
        print(f"\n  {split_name}")
        print(f"  {'Frac':>5}  {'VF':>6}  {'BToMI':>6}  {'GP':>6}  {'CTX':>6}  {'BNK':>6}")
        for f in EVAL_FRACTIONS:
            def _g(key):
                return results[key].get(split_name, {}).get(f, {}).get("top1", float("nan")) * 100
            print(f"  {f*100:>4.0f}%  {_g('VisitFrequency'):>5.1f}%  "
                  f"{_g('BToMInformed'):>5.1f}%  {_g('GoalPredictor'):>5.1f}%  "
                  f"{_g('ContextGoalPredictor'):>5.1f}%  {_g('BDIBottleneck'):>5.1f}%")
    print()

    # ── Save & plot ────────────────────────────────────────────────────
    # Serialise: convert float keys to strings for JSON
    json_results = {
        model: {
            split: {str(k): v for k, v in frac_dict.items()}
            for split, frac_dict in split_dict.items()
        }
        for model, split_dict in results.items()
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results → {results_path}")

    evaluated = set(args.splits)

    if "test" in evaluated or "test_new_agent" in evaluated:
        has_std = plot_top1_comparison_and_gap(
            results,
            split_test="test",
            split_new="test_new_agent",
            out_path_comparison=out_dir / "top1_comparison.png",
            out_path_gap=out_dir / "top1_comparison_gap.png",
            title_suffix="",
            paper=args.paper,
        )
        if not has_std:
            print("  NOTE: Missing one or both standard splits for top-1 comparison/gap plots.")
        else:
            plot_top1_combined(
                results,
                split_test="test",
                split_new="test_new_agent",
                out_path=out_dir / "top1_combined.png",
                title_suffix="",
                paper=args.paper,
            )

        if "test" in evaluated:
            plot_top5_split(results, "test", out_dir / "top5_test.png", paper=args.paper)
        if "test_new_agent" in evaluated:
            plot_top5_split(results, "test_new_agent", out_dir / "top5_test_new_agent.png", paper=args.paper)

    if "test_bh" in evaluated or "test_new_agent_bh" in evaluated:
        has_bh = plot_top1_comparison_and_gap(
            results,
            split_test="test_bh",
            split_new="test_new_agent_bh",
            out_path_comparison=out_dir / "top1_comparison_bh.png",
            out_path_gap=out_dir / "top1_comparison_gap_bh.png",
            title_suffix=" (BH)",
            paper=args.paper,
        )
        if not has_bh:
            print("  NOTE: Missing one or both BH splits for BH top-1 comparison/gap plots.")
        else:
            plot_top1_combined(
                results,
                split_test="test_bh",
                split_new="test_new_agent_bh",
                out_path=out_dir / "top1_combined_bh.png",
                title_suffix=" (BH)",
                paper=args.paper,
            )
            _generate_bh_table_tex(results, out_dir / "top1_bh_table.tex")


if __name__ == "__main__":
    main()
