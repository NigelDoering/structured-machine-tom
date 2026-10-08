"""
scripts/eval_counterfactual_desire.py

Desire-Transplant Counterfactual Experiment.

For each test episode we:
  1. Encode the desire context (K past trajectories) → desire_probs  ∈ Δ^{M-1}
     Encode the recent context (K_recent trajectories) → belief_probs ∈ [0,1]^M
     These are FRACTION-INDEPENDENT: they use full past-trajectory contexts.
  2. Encode the current trajectory truncated to a given path fraction → z_tau.
     This is re-computed at each fraction (25 / 50 / 75 / 100 %).
  3. Replace desire_probs with a category prototype — the mean desire_probs over
     all test episodes belonging to that target category.
  4. Re-run only the goal head with swapped desire and original belief_probs / z_tau.

Key insight
-----------
desire_probs and belief_probs depend only on past-trajectory context, not on the
current path length. Only z_tau changes with path fraction. This lets us encode
context once and sweep fractions cheaply by re-encoding z_tau only.

Results
-------
  Transplant matrix [6 × 6] at each fraction:
    Row  i = source episode's original desire category.
    Col  j = desire prototype swapped in.
    Cell   = mean P(goal ∈ POIs[j]).

  The diagonal should dominate when desire matters (early fractions).
  At 100 % the trajectory z_tau is highly informative and dominates goal
  prediction, so the diagonal advantage shrinks — showing that desire is
  most causally relevant early on.

Output
------
  results/counterfactual_desire/results.json
  results/counterfactual_desire/transplant_matrix_{split}.png   (one per split)

Usage
-----
  .venv/bin/python scripts/eval_counterfactual_desire.py
  .venv/bin/python scripts/eval_counterfactual_desire.py --fractions 0.25 0.5 0.75 1.0
    .venv/bin/python scripts/eval_counterfactual_desire.py --prototype-modes model handcrafted
  .venv/bin/python scripts/eval_counterfactual_desire.py --paper
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
CAT_LABELS = ["Attend\nClass", "Get\nFood", "Study", "Socialize", "Exercise", "Rest\nHome"]


# ── POI helpers ────────────────────────────────────────────────────────

def build_category_poi_indices(
    poi_cfg:    dict,
    poi_to_idx: dict[str, int],
) -> dict[str, list[int]]:
    return {
        cat: [poi_to_idx[p["id"]] for p in cat_data["pois"]]
        for cat, cat_data in poi_cfg["categories"].items()
    }


# ── Phase 1a — encode context components (fraction-independent) ───────

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
) -> list[dict[str, torch.Tensor]]:
    """
    Encode desire_probs and belief_probs for every episode.

    These use only the K past-trajectory (desire) context and K_recent
    (recent) context — both are full trajectories independent of the
    current path fraction.  z_tau is NOT computed here; it is computed
    separately per fraction in encode_z_tau_at_fraction().

    Returns a list of dicts with keys:
      desire_probs : (M,)
      belief_probs : (M,)
      desire_ctx   : (K,   T_d, 2) — stored for reuse in z_tau encoding
      desire_mask  : (K,   T_d)
      recent_ctx   : (K_r, T_r, 2)
      recent_mask  : (K_r, T_r)
    """
    model.eval()
    rng       = np.random.default_rng(rng_seed)
    agent_idx = _build_agent_index(records)
    flat_pos: dict[str, dict[int, int]] = {
        aid: {fi: pos for pos, fi in enumerate(idxs)}
        for aid, idxs in agent_idx.items()
    }

    def stack_context(ctx_list):
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

        dc, dm = stack_context(desire_ctx_list)
        rc, rm = stack_context(recent_ctx_list)

        # Run desire and recent encoders only — build a dummy 1-step current tensor
        # so encode_components() runs, then we discard z_tau here.
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
                # Store raw tensors for z_tau re-encoding
                "desire_ctx":   dc[i],   # (K,   T_d, 2)
                "desire_mask":  dm[i],   # (K,   T_d)
                "recent_ctx":   rc[i],   # (K_r, T_r, 2)
                "recent_mask":  rm[i],   # (K_r, T_r)
            }

    if any(c is None for c in components):
        raise RuntimeError("Internal error: failed to encode one or more episodes.")
    return [c for c in components if c is not None]


# ── Phase 1b — encode z_tau at a given path fraction ─────────────────

@torch.no_grad()
def encode_z_tau_at_fraction(
    model:       BDIBottleneckPredictor,
    records:     list[EpisodeRecord],
    components:  list[dict[str, torch.Tensor]],
    node_coords: dict,
    fraction:    float,
    device,
    batch_size:  int = 64,
) -> list[torch.Tensor]:
    """
    Re-encode z_tau for every episode using the current trajectory truncated
    to `fraction` of its length.  Reuses stored desire/recent context tensors
    so the encoders only run on the (small) current trajectory.

    Returns a list of (d_model,) tensors, one per episode.
    """
    model.eval()

    def stack_ctx_stored(comps, key_c, key_m):
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

        dc, dm = stack_ctx_stored(batch_comps, "desire_ctx",  "desire_mask")
        rc, rm = stack_ctx_stored(batch_comps, "recent_ctx",  "recent_mask")

        _, _, zt = model.encode_components(
            dc.to(device), dm.to(device),
            rc.to(device), rm.to(device),
            cc.to(device), cm.to(device),
        )
        zt = zt.cpu()

        for i in range(B):
            z_taus[start + i] = zt[i]

    return z_taus


# ── Phase 2 — desire prototypes ────────────────────────────────────────

def build_prototypes(
    components: list[dict[str, torch.Tensor]],
    records:    list[EpisodeRecord],
    poi_order_len: int,
) -> dict[str, torch.Tensor]:
    """
    Average desire_probs over all episodes in each category.
    Falls back to uniform if a category has no episodes.
    """
    acc: dict[str, list[torch.Tensor]] = defaultdict(list)
    for r, comp in zip(records, components):
        acc[r.desire].append(comp["desire_probs"])
    protos = {cat: torch.stack(vecs).mean(0) for cat, vecs in acc.items()}
    for cat in CATEGORIES:
        if cat not in protos:
            protos[cat] = torch.full((poi_order_len,), 1.0 / poi_order_len)
    return protos


def build_handcrafted_prototypes(
    cat_poi_indices: dict[str, list[int]],
    poi_order_len: int,
) -> dict[str, torch.Tensor]:
    """
    Handcrafted category-only prototypes.

    For each category c, assign all probability mass uniformly over POIs in c,
    and zero elsewhere.
    """
    protos: dict[str, torch.Tensor] = {}
    for cat in CATEGORIES:
        v = torch.zeros(poi_order_len, dtype=torch.float32)
        idx = cat_poi_indices.get(cat, [])
        if idx:
            v[idx] = 1.0 / len(idx)
        else:
            v[:] = 1.0 / poi_order_len
        protos[cat] = v
    return protos


# ── Phase 3 — counterfactual sweep at one fraction ────────────────────

@torch.no_grad()
def run_transplant_sweep(
    model:           BDIBottleneckPredictor,
    components:      list[dict[str, torch.Tensor]],
    z_taus:          list[torch.Tensor],
    records:         list[EpisodeRecord],
    prototypes:      dict[str, torch.Tensor],
    cat_poi_indices: dict[str, list[int]],
    device,
    batch_size:      int = 64,
) -> tuple[dict, dict]:
    """
    Sweep all 6 prototypes across all episodes, holding belief_probs and z_tau fixed.

    Returns
    -------
    sweep   : {d_orig: {d_tgt: [P(goal ∈ POIs[d_tgt]), ...]}}
    noswap  : {d_tgt:  [P(goal ∈ POIs[d_tgt]) under no swap, per episode]}
    """
    model.eval()
    n_cats = len(CATEGORIES)

    proto_tensor = torch.stack(
        [prototypes[d] for d in CATEGORIES]
    ).to(device)   # (6, M)

    # sweep[d_orig][d_tgt]  = P(goal ∈ POIs[d_tgt]) after swapping to prototype d_tgt
    # noswap[d_orig][d_tgt] = P(goal ∈ POIs[d_tgt]) with original desire (no swap)
    sweep:  dict[str, dict[str, list[float]]] = {d: {dt: [] for dt in CATEGORIES} for d in CATEGORIES}
    noswap: dict[str, dict[str, list[float]]] = {d: {dt: [] for dt in CATEGORIES} for d in CATEGORIES}

    for start in range(0, len(records), batch_size):
        batch_recs  = records[start : start + batch_size]
        batch_comps = components[start : start + batch_size]
        batch_zt    = z_taus[start : start + batch_size]
        B = len(batch_recs)

        dp = torch.stack([c["desire_probs"] for c in batch_comps]).to(device)  # (B, M)
        bp = torch.stack([c["belief_probs"] for c in batch_comps]).to(device)  # (B, M)
        zt = torch.stack(batch_zt).to(device)                                  # (B, D)

        # ── No-swap baseline — record P(goal ∈ POIs[d_tgt]) for ALL d_tgt ──
        base_probs = F.softmax(model.predict_goal(dp, bp, zt), dim=-1).cpu()   # (B, M)
        for i, r in enumerate(batch_recs):
            for d_tgt in CATEGORIES:
                idx = cat_poi_indices[d_tgt]
                noswap[r.desire][d_tgt].append(base_probs[i, idx].sum().item())

        # ── Counterfactual: all 6 prototypes in one batched call ───────
        bp_exp = bp.unsqueeze(1).expand(-1, n_cats, -1).reshape(B * n_cats, -1)
        zt_exp = zt.unsqueeze(1).expand(-1, n_cats, -1).reshape(B * n_cats, -1)
        dp_exp = proto_tensor.unsqueeze(0).expand(B, -1, -1).reshape(B * n_cats, -1)

        cf_probs = F.softmax(
            model.predict_goal(dp_exp, bp_exp, zt_exp), dim=-1
        ).cpu().view(B, n_cats, -1)   # (B, 6, M)

        for i, r in enumerate(batch_recs):
            for j, d_tgt in enumerate(CATEGORIES):
                idx = cat_poi_indices[d_tgt]
                sweep[r.desire][d_tgt].append(cf_probs[i, j, idx].sum().item())

    return sweep, noswap


# ── Aggregation ────────────────────────────────────────────────────────

def compute_matrix(sweep: dict) -> np.ndarray:
    """Raw probability mass matrix: cell[i,j] = mean P(goal ∈ POIs[j]) after swapping to prototype j."""
    mat = np.zeros((len(CATEGORIES), len(CATEGORIES)))
    for i, d_orig in enumerate(CATEGORIES):
        for j, d_tgt in enumerate(CATEGORIES):
            vals = sweep[d_orig][d_tgt]
            mat[i, j] = float(np.mean(vals)) if vals else 0.0
    return mat


def compute_delta_matrix(sweep: dict, noswap: dict) -> np.ndarray:
    """
    Δ probability mass matrix.

    delta[i,j] = mean P(goal ∈ POIs[j] | desire=prototype_j)
               − mean P(goal ∈ POIs[j] | original desire)

    For episodes where d_orig = i.

    Interpretation
    --------------
    Diagonal (i=j):   prototype matches own category → Δ ≈ 0 (no change expected).
    Off-diagonal:     positive Δ = transplant successfully increased probability mass
                      in the target category; negative = it decreased it.

    A well-structured model should show large positive Δ in off-diagonal cells where
    the transplanted desire is strongly different from the original (e.g. study→exercise).
    """
    delta = np.zeros((len(CATEGORIES), len(CATEGORIES)))
    for i, d_orig in enumerate(CATEGORIES):
        for j, d_tgt in enumerate(CATEGORIES):
            cf_vals   = sweep[d_orig][d_tgt]
            base_vals = noswap[d_orig][d_tgt]
            if cf_vals and base_vals:
                delta[i, j] = float(np.mean(cf_vals)) - float(np.mean(base_vals))
    return delta


# ── Significance testing ───────────────────────────────────────────────

def compute_desire_significance(
    sweep:   dict,
    noswap:  dict,
    records: list,
    n_perm:  int   = 10_000,
    n_boot:  int   = 5_000,
    seed:    int   = 42,
    alpha:   float = 0.05,
) -> dict:
    """
    Two-level significance tests for desire counterfactual interventions.

    Level 1 (aggregate): For each episode compute its mean off-diagonal delta
    (averaged across all 5 non-matching prototype targets), then test whether
    this pool of episode-level scalars has a positive mean.

    Level 2 (per-pair): For each of the 30 off-diagonal (source, target) pairs,
    run a one-sided sign test; apply Holm-Bonferroni correction across all 30.
    """
    # ── Level 1 ───────────────────────────────────────────────────────
    cat_counters = {cat: 0 for cat in CATEGORIES}
    episode_deltas: list[float] = []

    for r in records:
        d_orig = r.desire
        idx    = cat_counters[d_orig]
        cat_counters[d_orig] += 1

        offdiag = [
            sweep[d_orig][d_tgt][idx] - noswap[d_orig][d_tgt][idx]
            for d_tgt in CATEGORIES
            if d_tgt != d_orig
        ]
        episode_deltas.append(float(np.mean(offdiag)))

    deltas   = np.array(episode_deltas)
    n        = len(deltas)
    obs_mean = float(deltas.mean())

    # One-sided random-sign permutation test (H0: mean = 0)
    rng        = np.random.default_rng(seed)
    perm_count = sum(
        1 for _ in range(n_perm)
        if (rng.choice([-1, 1], size=n) * deltas).mean() >= obs_mean
    )
    perm_p = max(1, perm_count) / n_perm

    # One-sided exact sign test (H1: median > 0)
    n_pos  = int((deltas > 0).sum())
    sign_p = float(binomtest(n_pos, n, 0.5, alternative="greater").pvalue)

    # 95% bootstrap CI for the mean
    rng2       = np.random.default_rng(seed + 1)
    boot_means = [
        rng2.choice(deltas, size=n, replace=True).mean()
        for _ in range(n_boot)
    ]
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    # ── Level 2 ───────────────────────────────────────────────────────
    pair_pvals: list[float] = []
    for d_orig in CATEGORIES:
        for d_tgt in CATEGORIES:
            if d_orig == d_tgt:
                continue
            n_ep         = len(sweep[d_orig][d_tgt])
            pair_deltas  = np.array([
                sweep[d_orig][d_tgt][i] - noswap[d_orig][d_tgt][i]
                for i in range(n_ep)
            ])
            n_pos_pair = int((pair_deltas > 0).sum())
            pair_pvals.append(
                float(binomtest(n_pos_pair, n_ep, 0.5, alternative="greater").pvalue)
            )

    n_pairs    = len(pair_pvals)   # = 30
    ranked_idx = sorted(range(n_pairs), key=lambda i: pair_pvals[i])
    n_sig      = 0
    for k, i in enumerate(ranked_idx):
        if pair_pvals[i] <= alpha / (n_pairs - k):
            n_sig += 1
        else:
            break

    return {
        "n":                    n,
        "mean":                 obs_mean,
        "ci_lo":                ci_lo,
        "ci_hi":                ci_hi,
        "perm_p":               perm_p,
        "sign_p":               sign_p,
        "n_significant_pairs":  n_sig,
        "n_pairs":              n_pairs,
    }


def write_desire_significance_tex(
    sig_results: dict,   # {(split_name, mode): result_dict}
    out_path:    Path,
) -> None:
    """Write a LaTeX table of aggregate significance results for the paper."""
    SPLIT_LABELS = {
        "test":           "Test",
        "test_new_agent": "Test (new agent)",
    }
    MODE_LABELS = {
        "model":       "Model-derived",
        "handcrafted": "Handcrafted",
    }

    def fmt_perm_p(p: float) -> str:
        # perm_p is floored at 1/n_perm (= 1e-4 by default), so report it as a bound.
        if p <= 1e-4:
            return r"${\le}10^{-4}$"
        return f"${p:.4f}$"

    def fmt_sci(p: float) -> str:
        if p < 1e-3:
            mantissa, exp = f"{p:.1e}".split("e")
            return rf"${mantissa}\times10^{{{int(exp)}}}$"
        return f"${p:.4f}$"

    rows = []
    for (split, mode), res in sig_results.items():
        rows.append(
            f"    {SPLIT_LABELS.get(split, split):<16} & "
            f"{MODE_LABELS.get(mode, mode):<13} & "
            f"{res['n']} & "
            f"{res['mean']:.4f} & "
            f"[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}] & "
            f"{fmt_perm_p(res['perm_p'])} & "
            f"{fmt_sci(res['sign_p'])} & "
            f"{res['n_significant_pairs']}/{res['n_pairs']} \\\\"
        )

    tex = "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        (
            r"  \caption{Desire counterfactual significance at 50\% path reveal. "
            r"For each episode the mean off-diagonal $\Delta$ is computed as the "
            r"average change in goal probability mass toward the transplanted category "
            r"across all five non-matching prototype substitutions. "
            r"Permutation $p$: one-sided random-sign test (H$_0$: mean\,$=$\,0); "
            r"sign test $p$: one-sided exact test (H$_0$: median\,$=$\,0); "
            r"CI: 95\% bootstrap interval. "
            r"Pairs sig.: off-diagonal (source, target) pairs with mean $\Delta>0$ "
            r"significant at $\alpha=0.05$ after Holm--Bonferroni correction (30 pairs total).}"
        ),
        r"  \label{tab:cf_desire_significance}",
        r"  \begin{tabular}{llrrrrrr}",
        r"    \toprule",
        r"    Split & Prototypes & $n$ & Mean $\Delta$ & 95\% CI & Perm.\ $p$ & Sign test $p$ & Pairs sig. \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]) + "\n"

    out_path.write_text(tex)
    print(f"  Significance table → {out_path}")


# ── Plotting ───────────────────────────────────────────────────────────

def plot_prototype_distributions(
    prototypes:  dict[str, torch.Tensor],
    poi_cfg:     dict,
    prototype_label: str,
    out_path:    Path,
    paper:       bool = False,
) -> None:
    """
    6-panel bar chart — one panel per desire category.

    Each panel shows the full 24-POI desire_probs prototype for that category,
    with bars colored by POI category and grouped by category on the x-axis.
    A horizontal dashed line marks the uniform baseline (1/24 ≈ 0.042).

    This visualises whether the model's desire head has learned
    category-discriminative representations.  If the "study" prototype does
    NOT peak on study POIs, the desire head is not well calibrated.
    """
    cat_order  = list(poi_cfg["categories"].keys())
    poi_order  = [(p["id"], p["name"].split("(")[0].strip(), cat)
                  for cat in cat_order
                  for p in poi_cfg["categories"][cat]["pois"]]

    cat_colors = {
        "attend_class": "#5C6BC0",
        "get_food":     "#EF5350",
        "study":        "#26A69A",
        "socialize":    "#FFA726",
        "exercise":     "#66BB6A",
        "rest_home":    "#AB47BC",
    }
    cat_labels_clean = {
        "attend_class": "Attend Class",
        "get_food":     "Get Food",
        "study":        "Study",
        "socialize":    "Socialize",
        "exercise":     "Exercise",
        "rest_home":    "Rest / Home",
    }

    n_pois   = len(poi_order)
    uniform  = 1.0 / n_pois
    x        = np.arange(n_pois)
    bar_colors = [cat_colors[p[2]] for p in poi_order]
    short_names = [p[1][:10] for p in poi_order]

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=False)
    axes = axes.flatten()

    for ax, cat in zip(axes, CATEGORIES):
        proto = prototypes[cat].numpy()
        ax.bar(x, proto, color=bar_colors, alpha=0.85, width=0.7, zorder=3)
        ax.axhline(uniform, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.6, zorder=4)

        # Category group separators and labels on x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=8)

        # Shade category groups alternately for readability
        start_idx = 0
        for k, group_cat in enumerate(cat_order):
            n = len(poi_cfg["categories"][group_cat]["pois"])
            if k % 2 == 0:
                ax.axvspan(start_idx - 0.5, start_idx + n - 0.5,
                           color="gray", alpha=0.06, zorder=1)
            start_idx += n

        ax.set_title(f"Prototype: {cat_labels_clean[cat]}", fontsize=12,
                     fontweight="bold", color=cat_colors[cat], pad=12, y=1.02)
        ax.set_ylabel("Mean desire_probs", fontsize=11)
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{prototype_label} — Prototype Distributions by Category\n"
        "(each panel = prototype over 24 POIs)",
        fontsize=13, fontweight="bold", y=0.965,
    )

    # Single figure-level legend for the dashed uniform baseline.
    baseline_handle = Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
                             alpha=0.6, label=f"Uniform baseline (1/{n_pois} ≈ {uniform:.3f})")
    fig.legend(handles=[baseline_handle], loc="upper right", bbox_to_anchor=(0.99, 0.955),
               fontsize=9.5, frameon=False)

    dpi = 300 if paper else 150
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=2.0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_single_fraction_matrix(
    mat: np.ndarray,
    split_name: str,
    fraction: float,
    prototype_label: str,
    out_path: Path,
    paper: bool = False,
) -> None:
    """Single 6x6 delta heatmap for one split, one fraction, one prototype mode."""
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(mat, vmin=-0.3, vmax=0.3, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CAT_LABELS, fontsize=10)
    ax.set_yticklabels(CAT_LABELS, fontsize=10)
    ax.set_xlabel("Desire Transplanted In →", fontsize=11)
    ax.set_ylabel("Original Desire", fontsize=11)

    pct = int(fraction * 100)
    split_titles = {
        "test": "Test Set",
        "test_new_agent": "Test Set (New Agent)",
    }
    split_label = split_titles.get(split_name, split_name)
    ax.set_title(f"{split_label} — {pct}% path", fontsize=12, fontweight="bold")

    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            v = mat[i, j]
            color = "white" if abs(v) > 0.20 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    for k in range(len(CATEGORIES)):
        ax.add_patch(mpatches.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor="royalblue", linewidth=2.2,
        ))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ΔP(goal ∈ target cat.)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    if paper:
        fig.text(0.5, 1.03, f"{prototype_label}", ha="center", fontsize=12, fontweight="bold")
    else:
        fig.suptitle(f"{prototype_label}", fontsize=12, fontweight="bold", y=1.03)

    dpi = 300 if paper else 150
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=1.2)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_fraction_matrices(
    frac_matrices: dict[float, np.ndarray],
    split_name:    str,
    prototype_label: str,
    out_path:      Path,
    paper:         bool = False,
) -> None:
    """
    One row of heatmaps — one panel per path fraction.

    frac_matrices : {fraction: 6×6 ndarray}
    """
    fractions = sorted(frac_matrices.keys())
    n_fracs   = len(fractions)

    fig, axes = plt.subplots(1, n_fracs, figsize=(5.5 * n_fracs, 5.2))
    if n_fracs == 1:
        axes = [axes]

    split_titles = {
        "test":           "Test Set",
        "test_new_agent": "Test Set (New Agent)",
    }
    split_label = split_titles.get(split_name, split_name)

    for ax, frac in zip(axes, fractions):
        mat = frac_matrices[frac]
        im  = ax.imshow(mat, vmin=-0.3, vmax=0.3, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_xticklabels(CAT_LABELS, fontsize=10)
        ax.set_yticklabels(CAT_LABELS if frac == fractions[0] else [""] * len(CATEGORIES),
                           fontsize=10)
        ax.set_xlabel("Desire Transplanted In →", fontsize=11)
        if frac == fractions[0]:
            ax.set_ylabel("Original Desire", fontsize=11)

        pct = int(frac * 100)
        ax.set_title(f"{pct}% of path revealed", fontsize=12, fontweight="bold")

        # Cell annotations
        for i in range(len(CATEGORIES)):
            for j in range(len(CATEGORIES)):
                v     = mat[i, j]
                color = "white" if abs(v) > 0.20 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        # Diagonal boxes
        for k in range(len(CATEGORIES)):
            ax.add_patch(mpatches.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor="royalblue", linewidth=2.2,
            ))

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)
        if frac == fractions[-1]:
            cb.set_label("ΔP(goal ∈ target cat.)", fontsize=11)

    # Overall figure title
    title_str = f"Desire Transplant Δ Matrix — {split_label} — {prototype_label}"
    if paper:
        fig.text(0.5, 1.03, title_str, ha="center", fontsize=13, fontweight="bold")
    else:
        fig.suptitle(title_str, fontsize=13, fontweight="bold", y=1.03)

    dpi = 300 if paper else 150
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=1.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_diagonal_over_fractions(
    frac_diag: dict[str, dict[float, dict]],
    prototype_label: str,
    out_path:  Path,
    paper:     bool = False,
) -> None:
    """
    Line plot: diagonal mean vs path fraction, per split.

    Also shows off-diagonal mean to make the gap visible.
    This concisely answers "how much does desire matter at each path proportion?"
    """
    split_styles = {
        "test":           {"color": "#2196F3", "label": "Test",            "ls": "solid"},
        "test_new_agent": {"color": "#9C27B0", "label": "Test (new agent)","ls": "dashed"},
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for split_name, frac_data in frac_diag.items():
        style    = split_styles.get(split_name, {"color": "gray", "label": split_name, "ls": "solid"})
        fracs    = sorted(frac_data.keys())
        pct      = [int(f * 100) for f in fracs]
        diag     = [frac_data[f]["diag_mean"]    for f in fracs]
        offdiag  = [frac_data[f]["offdiag_mean"] for f in fracs]

        ax.plot(pct, diag, color=style["color"], linestyle=style["ls"],
                linewidth=2.2, marker="o", markersize=7,
                label=f"{style['label']} — diagonal (own desire)")
        ax.plot(pct, offdiag, color=style["color"], linestyle="dotted",
                linewidth=1.5, marker="s", markersize=5, alpha=0.65,
                label=f"{style['label']} — off-diagonal (other desire)")

    ax.set_xlabel("Path revealed (%)", fontsize=11)
    ax.set_ylabel("Mean P(goal ∈ target category)", fontsize=11)
    ax.set_xticks([int(f * 100) for f in sorted(next(iter(frac_diag.values())).keys())])
    ax.tick_params(labelsize=10)
    ax.set_ylim(0, 0.75)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9.5, loc="upper left")

    if not paper:
        ax.set_title(f"Desire Influence vs Path Fraction ({prototype_label})",
                     fontsize=13, fontweight="bold")

    dpi = 300 if paper else 150
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _print_matrix(mat: np.ndarray, records: list[EpisodeRecord], frac: float) -> None:
    pct = int(frac * 100)
    print(f"\n  Transplant matrix @ {pct}% path")
    header = "             " + "  ".join(f"{c[:7]:>7}" for c in CATEGORIES)
    print(header)
    for i, d_orig in enumerate(CATEGORIES):
        n_ep = sum(1 for r in records if r.desire == d_orig)
        row  = f"  {d_orig[:10]:<10} (n={n_ep:>4})  " + \
               "  ".join(f"{mat[i,j]:>7.3f}" for j in range(len(CATEGORIES)))
        print(row)
    diag    = float(np.mean([mat[i, i] for i in range(len(CATEGORIES))]))
    offdiag = float(np.mean([mat[i, j] for i in range(len(CATEGORIES))
                             for j in range(len(CATEGORIES)) if i != j]))
    print(f"  diag={diag:.3f}  off-diag={offdiag:.3f}  Δ={diag-offdiag:.3f}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Desire-transplant counterfactual experiment for BDIBottleneck."
    )
    p.add_argument("--bnk-ckpt",   default="outputs/bdi_bottleneck/best.pt", metavar="PT")
    p.add_argument("--splits-dir", default="data/splits",                    metavar="DIR")
    p.add_argument("--splits",     nargs="+", default=["test", "test_new_agent"])
    p.add_argument("--fractions",  nargs="+", type=float,
                   default=[0.25, 0.50, 0.75, 1.0],
                   metavar="F", help="Path fractions to evaluate (e.g. 0.25 0.5 0.75 1.0)")
    p.add_argument("--K",          type=int,  default=10, metavar="K")
    p.add_argument("--K-recent",   type=int,  default=5,  metavar="K", dest="K_recent")
    p.add_argument("--batch",      type=int,  default=64, metavar="N")
    p.add_argument(
        "--prototype-modes",
        nargs="+",
        choices=["model", "handcrafted"],
        default=["model", "handcrafted"],
        help="Prototype sources to evaluate in one run.",
    )
    p.add_argument("--no-cuda",    action="store_true")
    p.add_argument("--paper",      action="store_true",
                   help="300 DPI, no figure titles")
    return p.parse_args()


def _load_records(path: Path) -> list[EpisodeRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_json(line))
    return records


def _mode_label(mode: str) -> str:
    if mode == "model":
        return "Model-derived prototypes"
    if mode == "handcrafted":
        return "Handcrafted category-only prototypes"
    return mode


def main() -> None:
    args       = parse_args()
    splits_dir = ROOT / args.splits_dir
    out_dir    = ROOT / "results" / "counterfactual_desire"
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions  = sorted(args.fractions)

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
    print(f"Fractions: {[int(f*100) for f in fractions]}%\n")
    print(f"Prototype modes: {args.prototype_modes}\n")

    # ── POI config ────────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order       = [p["id"] for cd in poi_cfg["categories"].values() for p in cd["pois"]]
    poi_to_idx      = {pid: i for i, pid in enumerate(poi_order)}
    cat_poi_indices = build_category_poi_indices(poi_cfg, poi_to_idx)

    # ── Graph ─────────────────────────────────────────────────────────
    import osmnx as ox
    print("Loading graph …")
    G           = ox.load_graphml(ROOT / "data" / "ucsd_walk.graphml")
    node_coords = node_coords_from_graph(G)
    print(f"  {len(node_coords):,} nodes\n")

    # ── Load model ────────────────────────────────────────────────────
    bnk_path = ROOT / args.bnk_ckpt
    ckpt     = torch.load(bnk_path, map_location=device)
    state    = ckpt["model_state"]
    d_model  = state["coord_proj.weight"].shape[0]
    num_pois = state["desire_head.weight"].shape[0]
    model    = BDIBottleneckPredictor(num_pois=num_pois, d_model=d_model).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"BDIBottleneck loaded  ({sum(p.numel() for p in model.parameters()):,} params)\n")

    # ── Per-split evaluation ──────────────────────────────────────────
    json_results: dict[str, dict] = {}
    # frac_diag_by_mode[mode][split][fraction] = {diag_mean, offdiag_mean, ...}
    frac_diag_by_mode: dict[str, dict[str, dict[float, dict]]] = {
        mode: {} for mode in args.prototype_modes
    }
    # Significance results collected at fraction=0.50, written as one table at the end.
    # Key: (split_name, mode)
    sig_results_all: dict[tuple[str, str], dict] = {}

    for split_name in args.splits:
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping.")
            continue

        records = _load_records(path)
        print(f"{'='*60}")
        print(f"  Split: {split_name}  ({len(records)} episodes)")
        print(f"{'='*60}")

        # ── Phase 1a: encode context (once per split) ─────────────────
        print("  [1/3] Encoding context components (desire + belief) …")
        components = encode_context_components(
            model, records, node_coords, device,
            K=args.K, K_recent=args.K_recent, batch_size=args.batch,
        )

        # ── Phase 2/3 per prototype mode ─────────────────────────────
        json_results[split_name] = {}
        for mode in args.prototype_modes:
            print(f"\n  [2/3] Building desire prototypes ({mode}) …")
            if mode == "handcrafted":
                prototypes = build_handcrafted_prototypes(cat_poi_indices, len(poi_order))
            else:
                prototypes = build_prototypes(components, records, len(poi_order))

            for cat in CATEGORIES:
                n = sum(1 for r in records if r.desire == cat)
                print(f"    {cat:<18}  {n:>5} episodes  "
                      f"‖proto‖={prototypes[cat].norm().item():.3f}")

            print(f"\n  [3/3] Sweeping {len(fractions)} path fractions ({mode}) …")
            frac_matrices: dict[float, np.ndarray] = {}
            frac_diag_by_mode[mode].setdefault(split_name, {})
            json_split_mode: dict[str, dict] = {}

            for frac in fractions:
                pct = int(frac * 100)
                print(f"    Fraction {pct}% — encoding z_tau …", end=" ", flush=True)
                z_taus = encode_z_tau_at_fraction(
                    model, records, components, node_coords, frac, device, args.batch
                )
                print("running sweep …", end=" ", flush=True)
                sweep, noswap = run_transplant_sweep(
                    model, components, z_taus, records, prototypes,
                    cat_poi_indices, device, batch_size=args.batch,
                )
                mat       = compute_matrix(sweep)
                delta_mat = compute_delta_matrix(sweep, noswap)

                # Run significance tests at the primary operating point (50%).
                if abs(frac - 0.50) < 1e-6:
                    print("    Running significance tests …", end=" ", flush=True)
                    sig = compute_desire_significance(sweep, noswap, records)
                    sig_results_all[(split_name, mode)] = sig
                    print(
                        f"mean Δ={sig['mean']:.4f}  perm_p={sig['perm_p']:.4f}  "
                        f"sign_p={sig['sign_p']:.4e}  pairs_sig={sig['n_significant_pairs']}/{sig['n_pairs']}"
                    )
                frac_matrices[frac] = delta_mat
                _print_matrix(mat, records, frac)

                diag    = float(np.mean([mat[i, i] for i in range(len(CATEGORIES))]))
                offdiag = float(np.mean([mat[i, j] for i in range(len(CATEGORIES))
                                         for j in range(len(CATEGORIES)) if i != j]))
                delta_diag    = float(np.mean([delta_mat[i, i] for i in range(len(CATEGORIES))]))
                delta_offdiag = float(np.mean([delta_mat[i, j] for i in range(len(CATEGORIES))
                                               for j in range(len(CATEGORIES)) if i != j]))
                frac_diag_by_mode[mode][split_name][frac] = {
                    "diag_mean":          diag,
                    "offdiag_mean":       offdiag,
                    "delta_diag_mean":    delta_diag,
                    "delta_offdiag_mean": delta_offdiag,
                }

                json_split_mode[str(frac)] = {
                    "prototype_mode":          mode,
                    "transplant_matrix":       mat.tolist(),
                    "delta_matrix":            delta_mat.tolist(),
                    "diagonal_mean":           diag,
                    "offdiag_mean":            offdiag,
                    "delta_diagonal_mean":     delta_diag,
                    "delta_offdiag_mean":      delta_offdiag,
                    "per_cell_means": {
                        d_orig: {d_tgt: float(np.mean(sweep[d_orig][d_tgt]))
                                 for d_tgt in CATEGORIES}
                        for d_orig in CATEGORIES
                    },
                    "per_cell_deltas": {
                        d_orig: {d_tgt: float(np.mean(sweep[d_orig][d_tgt]))
                                        - float(np.mean(noswap[d_orig][d_tgt]))
                                 for d_tgt in CATEGORIES}
                        for d_orig in CATEGORIES
                    },
                }
                print("done.")

            json_results[split_name][mode] = json_split_mode

            suf = "_paper" if args.paper else ""
            mode_tag = mode
            mode_label = _mode_label(mode)

            pfname = f"desire_prototypes_{split_name}_{mode_tag}{suf}.png"
            plot_prototype_distributions(
                prototypes,
                poi_cfg,
                mode_label,
                out_dir / pfname,
                paper=args.paper,
            )

            fname = f"transplant_delta_{split_name}_{mode_tag}{suf}.png"
            plot_fraction_matrices(
                frac_matrices,
                split_name,
                mode_label,
                out_dir / fname,
                paper=args.paper,
            )

            # Additional single-fraction plots for manuscript main text.
            for frac in fractions:
                pct = int(frac * 100)
                sfname = f"transplant_delta_{split_name}_{mode_tag}_{pct}{suf}.png"
                plot_single_fraction_matrix(
                    frac_matrices[frac],
                    split_name,
                    frac,
                    mode_label,
                    out_dir / sfname,
                    paper=args.paper,
                )

    # ── Plot: diagonal vs fraction (all splits on one figure) ─────────
    for mode in args.prototype_modes:
        mode_frac_diag = {
            split: frac_data
            for split, frac_data in frac_diag_by_mode.get(mode, {}).items()
            if frac_data
        }
        if mode_frac_diag:
            fname2 = f"desire_influence_{mode}{'_paper' if args.paper else ''}.png"
            plot_diagonal_over_fractions(
                mode_frac_diag,
                _mode_label(mode),
                out_dir / fname2,
                paper=args.paper,
            )

    # ── Write significance table ───────────────────────────────────────
    if sig_results_all:
        # Reorder rows: test before test_new_agent, model before handcrafted.
        ordered_keys = [
            (split, mode)
            for split in args.splits
            for mode  in args.prototype_modes
            if (split, mode) in sig_results_all
        ]
        ordered_sig = {k: sig_results_all[k] for k in ordered_keys}
        write_desire_significance_tex(
            ordered_sig,
            out_dir / "significance_table.tex",
        )

    # ── Save JSON ─────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults   → {results_path}")
    print(f"All plots → {out_dir}/")


if __name__ == "__main__":
    main()
