"""
scripts/eval_counterfactual_belief.py

Belief counterfactual reasoning experiment.

Question: when we surgically remove the agent's belief that its intended goal
POI is open (while ensuring all other same-category POIs are believed open),
how does the model redistribute goal probability mass across the alternatives?

Experiment design
-----------------
1. Filter episodes to those where the travelled goal is within the agent's
    top-k preference ranks (default k=3).

2. At path fraction F (default 0.5), encode:
     desire_probs (M,)  — from past trajectory context
     belief_probs (M,)  — from past trajectory context
     z_tau        (d,)  — from current trajectory truncated at F

3. Construct the counterfactual belief vector:
     bp_cf = copy of belief_probs
     bp_cf[goal_poi_idx]              = 0.0   ("close" the intended goal)
     bp_cf[other same-category POIs]  = 1.0   ("open" the alternatives)

4. Run model.predict_goal() with:
     (a) original: desire_probs, belief_probs, z_tau
     (b) counterfactual: desire_probs, bp_cf, z_tau

5. For each alternative (non-goal) POI in the same category:
     Δ[poi] = P_cf(goal = poi) − P_orig(goal = poi)

If the belief head drives the goal head correctly, alternative gain should be
positive and goal drop should be negative.

Outputs
-------
  results/counterfactual_belief/alt_gain_{split}.png
      Mean alternative gain by category at the chosen fraction.

  results/counterfactual_belief/goal_drop_{split}.png
      Mean goal-probability drop by category at the chosen fraction.

  results/counterfactual_belief/results.json

Usage
-----
    .venv/bin/python scripts/eval_counterfactual_belief.py
    .venv/bin/python scripts/eval_counterfactual_belief.py --top-k 3 --fractions 0.25 0.5 0.75 1.0 --paper
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model.bdi_bottleneck import BDIBottleneckPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.meta_dataset import _norm
from simulation.episode import EpisodeRecord
from scripts.train_bdi_predictor import _build_agent_index, _build_context_tensor


CATEGORIES = ["attend_class", "get_food", "study", "socialize", "exercise", "rest_home"]

CATEGORY_COLORS: dict[str, str] = {
    "attend_class": "#5C6BC0",
    "get_food":     "#EF5350",
    "study":        "#26A69A",
    "socialize":    "#FFA726",
    "exercise":     "#66BB6A",
    "rest_home":    "#AB47BC",
}

CATEGORY_LABELS: dict[str, str] = {
    "attend_class": "Attend Class",
    "get_food":     "Get Food",
    "study":        "Study",
    "socialize":    "Socialize",
    "exercise":     "Exercise",
    "rest_home":    "Rest / Home",
}

SPLIT_LABELS: dict[str, str] = {
    "test":           "Test",
    "test_new_agent": "New Agent",
}

# ── Helpers ────────────────────────────────────────────────────────────

def _load_records(path: Path) -> list[EpisodeRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_json(line))
    return records


def _pref_rank(desire_vector: list[float], poi_idx: int) -> int:
    """Rank of poi_idx in desire_vector; rank 1 = most preferred."""
    dv = np.array(desire_vector)
    sorted_idx = np.argsort(dv)[::-1]
    return int(np.where(sorted_idx == poi_idx)[0][0]) + 1


def _filter_top_k(
    records: list[EpisodeRecord],
    poi_to_idx: dict[str, int],
    top_k: int,
) -> list[EpisodeRecord]:
    """Keep episodes where the travelled goal POI rank is <= top_k."""
    filtered = []
    for r in records:
        poi_idx = poi_to_idx[r.goal_poi]
        if _pref_rank(r.desire_vector, poi_idx) <= top_k:
            filtered.append(r)
    return filtered


# ── Phase 1 — encode context components (desire/belief heads) ──────────

def _stack_context(ctx_list):
    K_  = ctx_list[0][0].size(0)
    T_  = max(c.size(1) for c, _ in ctx_list)
    B_  = len(ctx_list)
    co  = torch.zeros(B_, K_, T_, 2)
    ma  = torch.zeros(B_, K_, T_, dtype=torch.long)
    for i, (c, m) in enumerate(ctx_list):
        t = c.size(1)
        co[i, :, :t, :] = c
        ma[i, :, :t]    = m
    return co, ma


@torch.no_grad()
def encode_context_components(
    model:       BDIBottleneckPredictor,
    records:     list[EpisodeRecord],
    node_coords: dict,
    device,
    K:           int = 10,
    K_recent:    int = 5,
    batch_size:  int = 64,
    rng_seed:    int = 42,
) -> list[dict]:
    """
    Encode desire_probs, belief_probs and store raw context tensors for
    later z_tau re-encoding.  Fraction-independent.

    Returns list of dicts with:
      desire_probs, belief_probs,
      desire_ctx, desire_mask, recent_ctx, recent_mask
    """
    model.eval()
    rng       = np.random.default_rng(rng_seed)
    agent_idx = _build_agent_index(records)
    flat_pos: dict[str, dict[int, int]] = {
        aid: {fi: pos for pos, fi in enumerate(idxs)}
        for aid, idxs in agent_idx.items()
    }

    # We need to operate on a flat index within the full records list, but
    # records here is already the filtered subset.  Rebuild flat_pos for subset.
    # Because _build_agent_index uses the order it receives, indices are local.
    components: list[dict | None] = [None] * len(records)

    for start in range(0, len(records), batch_size):
        batch_recs = records[start : start + batch_size]
        B = len(batch_recs)

        desire_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        recent_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []

        for flat_idx, r in enumerate(batch_recs, start=start):
            agent_idxs = agent_idx[r.agent_id]
            pos        = flat_pos[r.agent_id][flat_idx]
            others     = [i for i in agent_idxs if i != flat_idx]

            if len(others) == 0:
                d_src = [flat_idx] * K
            elif len(others) < K:
                d_src = rng.choice(others, size=K, replace=True).tolist()
            else:
                d_src = rng.choice(others, size=K, replace=False).tolist()

            preceding = agent_idxs[:pos]
            if len(preceding) == 0:
                fallback = agent_idxs[pos + 1:] or [flat_idx]
                r_src = (fallback * ((K_recent // len(fallback)) + 1))[:K_recent]
            elif len(preceding) < K_recent:
                r_src = (preceding * ((K_recent // len(preceding)) + 1))[:K_recent]
            else:
                r_src = preceding[-K_recent:]

            desire_ctx_list.append(
                _build_context_tensor([records[i] for i in d_src], node_coords, K)
            )
            recent_ctx_list.append(
                _build_context_tensor([records[i] for i in r_src], node_coords, K_recent)
            )

        dc, dm = _stack_context(desire_ctx_list)
        rc, rm = _stack_context(recent_ctx_list)

        dummy_cc = torch.zeros(B, 1, 2)
        dummy_cm = torch.ones(B, 1, dtype=torch.long)

        dp, bp, _ = model.encode_components(
            dc.to(device), dm.to(device),
            rc.to(device), rm.to(device),
            dummy_cc.to(device), dummy_cm.to(device),
        )
        dp, bp = dp.cpu(), bp.cpu()

        for i in range(B):
            components[start + i] = {
                "desire_probs": dp[i],
                "belief_probs": bp[i],
                "desire_ctx":   dc[i],
                "desire_mask":  dm[i],
                "recent_ctx":   rc[i],
                "recent_mask":  rm[i],
            }

    if any(c is None for c in components):
        raise RuntimeError("Internal error: failed to encode one or more episodes.")
    return [c for c in components]  # type: ignore[return-value]


# ── Phase 2 — encode z_tau at a given path fraction ───────────────────

@torch.no_grad()
def encode_z_tau_at_fraction(
    model:       BDIBottleneckPredictor,
    records:     list[EpisodeRecord],
    components:  list[dict],
    node_coords: dict,
    fraction:    float,
    device,
    batch_size:  int = 64,
) -> list[torch.Tensor]:
    """
    Encode z_tau for the current trajectory truncated to `fraction` of its
    length.  Reuses stored desire/recent context tensors.
    """
    model.eval()

    def stack_stored(comps, key_c, key_m):
        K_  = comps[0][key_c].size(0)
        T_  = max(c[key_c].size(1) for c in comps)
        B_  = len(comps)
        co  = torch.zeros(B_, K_, T_, 2)
        ma  = torch.zeros(B_, K_, T_, dtype=torch.long)
        for i, c in enumerate(comps):
            t = c[key_c].size(1)
            co[i, :, :t, :] = c[key_c]
            ma[i, :, :t]    = c[key_m]
        return co, ma

    z_taus: list[torch.Tensor] = [torch.empty(0)] * len(records)

    for start in range(0, len(records), batch_size):
        batch_recs  = records[start : start + batch_size]
        batch_comps = components[start : start + batch_size]
        B = len(batch_recs)

        curr_list: list[torch.Tensor] = []
        for r in batch_recs:
            T   = len(r.path)
            k   = max(1, round(fraction * T))
            pts = [_norm(*node_coords.get(nid, (0.0, 0.0))) for nid in r.path[:k]]
            curr_list.append(torch.tensor(pts, dtype=torch.float32))

        T_c = max(c.size(0) for c in curr_list)
        cc  = torch.zeros(B, T_c, 2)
        cm  = torch.zeros(B, T_c, dtype=torch.long)
        for i, c in enumerate(curr_list):
            cc[i, :c.size(0)] = c
            cm[i, :c.size(0)] = 1

        dc, dm = stack_stored(batch_comps, "desire_ctx",  "desire_mask")
        rc, rm = stack_stored(batch_comps, "recent_ctx",  "recent_mask")

        _, _, zt = model.encode_components(
            dc.to(device), dm.to(device),
            rc.to(device), rm.to(device),
            cc.to(device), cm.to(device),
        )
        zt = zt.cpu()

        for i in range(B):
            z_taus[start + i] = zt[i]

    return z_taus


# ── Phase 3 — belief counterfactual sweep ─────────────────────────────

@torch.no_grad()
def run_belief_counterfactual(
    model:            BDIBottleneckPredictor,
    records:          list[EpisodeRecord],
    components:       list[dict],
    z_taus:           list[torch.Tensor],
    poi_to_idx:       dict[str, int],
    poi_order:        list[str],
    cat_poi_indices:  dict[str, list[int]],
    device,
    batch_size:       int = 64,
) -> list[dict]:
    """
    For each episode, compute goal probabilities before and after the belief
    intervention, and return per-episode Δ for each alternative same-category POI.

    Intervention:
      bp_cf[goal_poi_idx]              = 0.0  (close intended goal)
      bp_cf[other same-category POIs]  = 1.0  (open alternatives)
      all other POIs: unchanged

    Returns list of dicts with:
            agent_id    : str
      goal_poi    : str
      desire_cat  : str
      alt_pois    : list[str] — other same-category POIs
      delta       : list[float] — Δ for each alt POI (cf − orig)
      p_orig_alt  : list[float] — original probability for each alt POI
      p_cf_alt    : list[float] — counterfactual probability for each alt POI
      p_orig_goal : float — original P(goal = goal_poi)
      p_cf_goal   : float — counterfactual P(goal = goal_poi)
    """
    model.eval()
    results: list[dict] = []

    for start in range(0, len(records), batch_size):
        batch_recs  = records[start : start + batch_size]
        batch_comps = components[start : start + batch_size]
        batch_zt    = z_taus[start : start + batch_size]
        B = len(batch_recs)

        # Build batch tensors for original inputs
        dp_batch = torch.stack([c["desire_probs"] for c in batch_comps])  # (B, M)
        bp_batch = torch.stack([c["belief_probs"] for c in batch_comps])  # (B, M)
        zt_batch = torch.stack(batch_zt)                                   # (B, d)

        # Build counterfactual belief batch
        bp_cf_batch = bp_batch.clone()
        goal_indices: list[int] = []
        alt_indices_list: list[list[int]] = []
        for i, r in enumerate(batch_recs):
            goal_idx = poi_to_idx[r.goal_poi]
            cat = r.desire
            same_cat = cat_poi_indices[cat]
            alt_idxs = [j for j in same_cat if j != goal_idx]

            bp_cf_batch[i, goal_idx] = 0.0
            for j in alt_idxs:
                bp_cf_batch[i, j] = 1.0

            goal_indices.append(goal_idx)
            alt_indices_list.append(alt_idxs)

        # Run goal head for original and counterfactual
        logits_orig = model.predict_goal(
            dp_batch.to(device), bp_batch.to(device), zt_batch.to(device)
        ).cpu()
        logits_cf = model.predict_goal(
            dp_batch.to(device), bp_cf_batch.to(device), zt_batch.to(device)
        ).cpu()

        p_orig = F.softmax(logits_orig, dim=-1)  # (B, M)
        p_cf   = F.softmax(logits_cf,   dim=-1)  # (B, M)

        for i, r in enumerate(batch_recs):
            goal_idx  = goal_indices[i]
            alt_idxs  = alt_indices_list[i]
            cat       = r.desire
            same_cat  = cat_poi_indices[cat]
            alt_pois  = [p for p, idx in poi_to_idx.items() if idx in alt_idxs]
            # Preserve order matching cat_poi_indices
            alt_pois_ordered  = [poi_order[j] for j in sorted(alt_idxs)]
            alt_idxs_ordered  = sorted(alt_idxs)

            delta     = [(p_cf[i, j] - p_orig[i, j]).item()  for j in alt_idxs_ordered]
            p_orig_alt = [p_orig[i, j].item() for j in alt_idxs_ordered]
            p_cf_alt   = [p_cf[i, j].item()   for j in alt_idxs_ordered]

            results.append({
                "agent_id":    r.agent_id,
                "goal_poi":    r.goal_poi,
                "desire_cat":  cat,
                "alt_pois":    alt_pois_ordered,
                "delta":       delta,
                "p_orig_alt":  p_orig_alt,
                "p_cf_alt":    p_cf_alt,
                "p_orig_goal": p_orig[i, goal_idx].item(),
                "p_cf_goal":   p_cf[i, goal_idx].item(),
            })

    return results


# ── Plotting ───────────────────────────────────────────────────────────

def _aggregate_fraction_metrics(cf_results: list[dict]) -> dict[str, dict[str, float | int | None]]:
    """
    Aggregate per-category metrics for one fraction.

    Returns
    -------
    {category: {
        n_episodes,
        alt_gain_mean, alt_gain_se,
        goal_drop_mean, goal_drop_se,
    }}
    """
    cat_alt_gain: dict[str, list[float]] = defaultdict(list)
    cat_goal_drop: dict[str, list[float]] = defaultdict(list)
    cat_n_eps: dict[str, int] = defaultdict(int)

    for ep in cf_results:
        cat = ep["desire_cat"]
        cat_n_eps[cat] += 1
        cat_alt_gain[cat].append(float(sum(ep["delta"])))
        cat_goal_drop[cat].append(float(ep["p_cf_goal"] - ep["p_orig_goal"]))

    out: dict[str, dict[str, float | int | None]] = {}
    for cat in CATEGORIES:
        n = len(cat_alt_gain.get(cat, []))
        if n == 0:
            out[cat] = {
                "n_episodes": 0,
                "alt_gain_mean": None,
                "alt_gain_se": None,
                "goal_drop_mean": None,
                "goal_drop_se": None,
            }
            continue

        alt_vals = np.array(cat_alt_gain[cat], dtype=np.float64)
        goal_vals = np.array(cat_goal_drop[cat], dtype=np.float64)
        out[cat] = {
            "n_episodes": int(cat_n_eps[cat]),
            "alt_gain_mean": float(np.mean(alt_vals)),
            "alt_gain_se": float(np.std(alt_vals, ddof=0) / np.sqrt(max(1, len(alt_vals)))),
            "goal_drop_mean": float(np.mean(goal_vals)),
            "goal_drop_se": float(np.std(goal_vals, ddof=0) / np.sqrt(max(1, len(goal_vals)))),
        }
    return out


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = values[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def _permutation_pvalue_mean_zero(
    values: np.ndarray,
    alternative: str,
    n_perm: int = 20000,
    seed: int = 11,
) -> float:
    """
    One-sample random-sign permutation test for mean(values) against zero.

    alternative:
      "greater" tests mean > 0
      "less"    tests mean < 0
    """
    if values.size == 0:
        return float("nan")
    obs = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, values.size), replace=True)
    null_means = (signs * values).mean(axis=1)

    if alternative == "greater":
        extreme = np.sum(null_means >= obs)
    elif alternative == "less":
        extreme = np.sum(null_means <= obs)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float((extreme + 1) / (n_perm + 1))


def _sign_test_zero(values: np.ndarray, alternative: str) -> dict[str, float | int | None]:
    """
    Exact one-sided sign test around zero.

    alternative:
      "greater" tests that values are more often > 0 than < 0
      "less"    tests that values are more often < 0 than > 0
    """
    pos = int(np.sum(values > 0))
    neg = int(np.sum(values < 0))
    n = pos + neg
    if n == 0:
        return {"n_nonzero": 0, "n_positive": 0, "n_negative": 0, "p_value": None}

    if alternative == "greater":
        k = pos
        tail = sum(math.comb(n, i) for i in range(k, n + 1))
    elif alternative == "less":
        k = neg
        tail = sum(math.comb(n, i) for i in range(k, n + 1))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    p = float(tail * (0.5 ** n))
    return {
        "n_nonzero": n,
        "n_positive": pos,
        "n_negative": neg,
        "p_value": min(1.0, p),
    }


def _cohen_dz(values: np.ndarray) -> float | None:
    """Paired-effect size for one-sample differences against zero."""
    if values.size < 2:
        return None
    sd = float(values.std(ddof=1))
    if sd == 0.0:
        return None
    return float(values.mean() / sd)


def _compute_significance(
    alt_gain: list[float],
    goal_drop: list[float],
) -> dict[str, dict[str, float | int | None | dict[str, float | int | None]]]:
    """Compute split-level significance diagnostics for belief counterfactual outcomes."""
    alt = np.array(alt_gain, dtype=np.float64)
    goal = np.array(goal_drop, dtype=np.float64)

    alt_ci = _bootstrap_mean_ci(alt)
    goal_ci = _bootstrap_mean_ci(goal)

    return {
        "alternative_gain": {
            "hypothesis": "H1: mean(alt_gain) > 0",
            "mean": float(alt.mean()) if alt.size else None,
            "std": float(alt.std()) if alt.size else None,
            "ci95_low": alt_ci[0],
            "ci95_high": alt_ci[1],
            "cohen_dz": _cohen_dz(alt),
            "permutation_p_one_sided": _permutation_pvalue_mean_zero(alt, "greater"),
            "sign_test": _sign_test_zero(alt, "greater"),
            "pct_correct_direction": float(np.mean(alt > 0)) if alt.size else None,
        },
        "goal_drop": {
            "hypothesis": "H1: mean(goal_drop) < 0",
            "mean": float(goal.mean()) if goal.size else None,
            "std": float(goal.std()) if goal.size else None,
            "ci95_low": goal_ci[0],
            "ci95_high": goal_ci[1],
            "cohen_dz": _cohen_dz(goal),
            "permutation_p_one_sided": _permutation_pvalue_mean_zero(goal, "less"),
            "sign_test": _sign_test_zero(goal, "less"),
            "pct_correct_direction": float(np.mean(goal < 0)) if goal.size else None,
        },
    }


def _aggregate_agent_level_effects(
    cf_results: list[dict],
) -> tuple[list[float], list[float], list[int]]:
    """
    Collapse episode-level deltas to one mean effect per agent.

    Returns
    -------
    alt_agent_means: episode alt-gain means per agent
    goal_agent_means: episode goal-drop means per agent
    episode_counts: number of episodes contributing to each agent mean
    """
    by_agent_alt: dict[str, list[float]] = defaultdict(list)
    by_agent_goal: dict[str, list[float]] = defaultdict(list)

    for ep in cf_results:
        agent_id = str(ep.get("agent_id", ""))
        if not agent_id:
            continue
        by_agent_alt[agent_id].append(float(sum(ep["delta"])))
        by_agent_goal[agent_id].append(float(ep["p_cf_goal"] - ep["p_orig_goal"]))

    if not by_agent_alt:
        return [], [], []

    agent_ids = sorted(by_agent_alt.keys())
    alt_agent_means = [float(np.mean(by_agent_alt[aid])) for aid in agent_ids]
    goal_agent_means = [float(np.mean(by_agent_goal[aid])) for aid in agent_ids]
    episode_counts = [len(by_agent_alt[aid]) for aid in agent_ids]
    return alt_agent_means, goal_agent_means, episode_counts


def _plot_metric_single_fraction(
    metrics: dict[str, dict[str, float | int | None]],
    split_name: str,
    out_path: Path,
    metric_key: str,
    se_key: str,
    ylabel: str,
    title_prefix: str,
    top_k: int,
    fraction: float,
    paper: bool = False,
) -> None:
    """Single-fraction category-level bar chart using canonical category colors."""
    cats = CATEGORIES
    x = np.arange(len(cats))

    vals: list[float] = []
    ses: list[float] = []
    for c in cats:
        mv = metrics[c][metric_key]
        sv = metrics[c][se_key]
        vals.append(float(mv) if isinstance(mv, (float, int)) else 0.0)
        ses.append(float(sv) if isinstance(sv, (float, int)) else 0.0)
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.bar(
        x,
        vals,
        yerr=ses,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.9,
        zorder=3,
    )

    # Place annotation beyond the error bar cap with a small clearance gap
    ann_gap = 0.004
    ann_ys: list[float] = []
    for xi, (v, se) in enumerate(zip(vals, ses)):
        va = "bottom" if v >= 0 else "top"
        ann_y = (v + se + ann_gap) if v >= 0 else (v - se - ann_gap)
        ax.text(xi, ann_y, f"{v:+.3f}", ha="center", va=va, fontsize=11)
        ann_ys.append(ann_y)

    # Expand y-axis so annotations are never clipped
    all_tops    = [v + se for v, se in zip(vals, ses)] + ann_ys
    all_bottoms = [v - se for v, se in zip(vals, ses)] + ann_ys
    y_span = max(all_tops) - min(all_bottoms)
    y_pad  = 0.12 * y_span
    ax.set_ylim(min(all_bottoms) - y_pad, max(all_tops) + y_pad)

    ax.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--", zorder=2)
    split_label = SPLIT_LABELS.get(split_name, split_name)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=12, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(
        f"{title_prefix} — {split_label} ({int(fraction*100)}% path)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    dpi = 300 if paper else 150
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Belief counterfactual: close intended goal, measure Δ on same-category alternatives."
    )
    p.add_argument("--bnk-ckpt",   default="outputs/bdi_bottleneck/best.pt", metavar="PT")
    p.add_argument("--splits-dir", default="data/splits",                    metavar="DIR")
    p.add_argument("--splits",     nargs="+", default=["test", "test_new_agent"])
    p.add_argument("--fractions",  type=float, nargs="+",
                   default=[0.25, 0.5, 0.75, 1.0], metavar="F",
                   help="Path fractions at which to encode z_tau.")
    p.add_argument("--fraction",   type=float, default=None, metavar="F",
                   help="Deprecated single-fraction alias. If set, overrides --fractions.")
    p.add_argument("--top-k",      type=int, default=3, metavar="K",
                   help="Keep episodes where travelled goal has preference rank <= top-k.")
    p.add_argument("--K",          type=int,  default=10, metavar="K")
    p.add_argument("--K-recent",   type=int,  default=5,  metavar="K", dest="K_recent")
    p.add_argument("--batch",      type=int,  default=64, metavar="N")
    p.add_argument("--no-cuda",    action="store_true")
    p.add_argument("--paper",      action="store_true", help="300 DPI output")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    splits_dir = ROOT / args.splits_dir
    out_dir    = ROOT / "results" / "counterfactual_belief"
    out_dir.mkdir(parents=True, exist_ok=True)

    fractions = [args.fraction] if args.fraction is not None else args.fractions
    if not fractions:
        print("ERROR: Provide at least one fraction via --fractions.")
        sys.exit(1)
    for f in fractions:
        if not (0.0 < float(f) <= 1.0):
            print(f"ERROR: invalid fraction {f}; each fraction must satisfy 0 < f <= 1.")
            sys.exit(1)
    # De-duplicate and sort for stable output names/order.
    fractions = sorted(set(float(f) for f in fractions))

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

    # Category → list of POI indices
    cat_poi_indices: dict[str, list[int]] = {}
    for cat, cat_data in poi_cfg["categories"].items():
        cat_poi_indices[cat] = [poi_to_idx[p["id"]] for p in cat_data["pois"]]

    # ── Graph ─────────────────────────────────────────────────────────
    import osmnx as ox
    print("Loading graph …")
    G           = ox.load_graphml(ROOT / "data" / "ucsd_walk.graphml")
    node_coords = node_coords_from_graph(G)
    print(f"  {len(node_coords):,} nodes\n")

    # ── Load model ────────────────────────────────────────────────────
    ckpt    = torch.load(ROOT / args.bnk_ckpt, map_location=device)
    state   = ckpt["model_state"]
    d_model = state["coord_proj.weight"].shape[0]
    num_pois = state["desire_head.weight"].shape[0]
    model   = BDIBottleneckPredictor(num_pois=num_pois, d_model=d_model).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"BDIBottleneck loaded  ({sum(p.numel() for p in model.parameters()):,} params)\n")

    all_json: dict[str, dict] = {}

    for split_name in args.splits:
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping.")
            continue

        records = _load_records(path)
        print(f"{'='*60}")
        print(f"  Split: {split_name}  ({len(records)} episodes total)")

        # Filter to top-k episodes
        rankk_records = _filter_top_k(records, poi_to_idx, args.top_k)
        print(
            f"  Rank<= {args.top_k} episodes: {len(rankk_records)} "
            f"({len(rankk_records)/len(records)*100:.1f}%)"
        )

        if len(rankk_records) == 0:
            print("  No episodes satisfy rank filter — skipping split.")
            continue

        # Category breakdown
        cat_counts = defaultdict(int)
        for r in rankk_records:
            cat_counts[r.desire] += 1
        for cat in CATEGORIES:
            if cat_counts[cat]:
                print(f"    {cat:<18} {cat_counts[cat]:>4} episodes")
        print()

        # ── Encode components (fraction-independent) ────────────────
        print(f"  Encoding context components …")
        components = encode_context_components(
            model, rankk_records, node_coords, device,
            K=args.K, K_recent=args.K_recent, batch_size=args.batch,
        )

        split_json: dict[str, dict] = {
            "top_k": args.top_k,
            "n_total_episodes": len(records),
            "n_filtered_episodes": len(rankk_records),
            "fractions": {},
        }

        for frac in fractions:
            print(f"  Encoding z_tau at fraction={frac:.2f} …")
            z_taus = encode_z_tau_at_fraction(
                model, rankk_records, components, node_coords,
                fraction=frac, device=device, batch_size=args.batch,
            )

            print("  Running belief counterfactual …")
            cf_results = run_belief_counterfactual(
                model, rankk_records, components, z_taus,
                poi_to_idx, poi_order, cat_poi_indices, device, batch_size=args.batch,
            )

            all_alt_gain = [float(sum(ep["delta"])) for ep in cf_results]
            all_goal_drop = [float(ep["p_cf_goal"] - ep["p_orig_goal"]) for ep in cf_results]
            alt_agent_means, goal_agent_means, agent_episode_counts = _aggregate_agent_level_effects(cf_results)
            metrics = _aggregate_fraction_metrics(cf_results)
            significance = _compute_significance(all_alt_gain, all_goal_drop)
            significance_clustered = _compute_significance(alt_agent_means, goal_agent_means)

            print(
                f"    mean alt gain={float(np.mean(all_alt_gain)):+.4f}  "
                f"mean goal drop={float(np.mean(all_goal_drop)):+.4f}"
            )
            print(
                "    one-sided permutation p-values: "
                f"alt_gain={significance['alternative_gain']['permutation_p_one_sided']:.4g}, "
                f"goal_drop={significance['goal_drop']['permutation_p_one_sided']:.4g}"
            )
            if alt_agent_means:
                print(
                    "    agent-clustered one-sided permutation p-values: "
                    f"alt_gain={significance_clustered['alternative_gain']['permutation_p_one_sided']:.4g}, "
                    f"goal_drop={significance_clustered['goal_drop']['permutation_p_one_sided']:.4g}"
                )

            frac_tag = int(round(frac * 100))
            print("\n  Saving plots …")
            _plot_metric_single_fraction(
                metrics,
                split_name,
                out_dir / f"alt_gain_{split_name}_f{frac_tag:03d}.png",
                metric_key="alt_gain_mean",
                se_key="alt_gain_se",
                ylabel="Mean alternative gain",
                title_prefix="Alternative Gain",
                top_k=args.top_k,
                fraction=frac,
                paper=args.paper,
            )
            _plot_metric_single_fraction(
                metrics,
                split_name,
                out_dir / f"goal_drop_{split_name}_f{frac_tag:03d}.png",
                metric_key="goal_drop_mean",
                se_key="goal_drop_se",
                ylabel="Mean goal prob. change (CF − orig.)",
                title_prefix="Goal Drop",
                top_k=args.top_k,
                fraction=frac,
                paper=args.paper,
            )

            split_json["fractions"][f"{frac:.2f}"] = {
                "fraction": frac,
                "n_episodes": len(cf_results),
                "mean_alt_gain": float(np.mean(all_alt_gain)) if all_alt_gain else None,
                "std_alt_gain": float(np.std(all_alt_gain)) if all_alt_gain else None,
                "mean_goal_drop": float(np.mean(all_goal_drop)) if all_goal_drop else None,
                "std_goal_drop": float(np.std(all_goal_drop)) if all_goal_drop else None,
                "per_category": metrics,
                "significance": significance,
                "significance_clustered_by_agent": {
                    "n_agents": len(agent_episode_counts),
                    "mean_episodes_per_agent": float(np.mean(agent_episode_counts)) if agent_episode_counts else None,
                    "min_episodes_per_agent": int(np.min(agent_episode_counts)) if agent_episode_counts else None,
                    "max_episodes_per_agent": int(np.max(agent_episode_counts)) if agent_episode_counts else None,
                    "alternative_gain": significance_clustered["alternative_gain"],
                    "goal_drop": significance_clustered["goal_drop"],
                },
            }

        all_json[split_name] = split_json

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_json, f, indent=2)
    print(f"\n  Results JSON → {results_path}")


if __name__ == "__main__":
    main()
