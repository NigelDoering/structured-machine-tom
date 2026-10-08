"""
scripts/eval_belief_evidence.py

Belief-head sensitivity as a function of agent-specific evidence.

Core idea
---------
For each (agent, Variable-tier POI) pair, we compute:

  evidence(i, j) = number of training episodes in which agent i's goal_poi == j
                   (how much behavioural signal the model has seen about this POI)

    Δ(i, j) = mean P̂(believed closed | agent believes j closed, b_j=0)
                     − mean P̂(believed closed | agent believes j open,   b_j=1)

             where P̂(believed closed | episode) = 1 − belief_probs[j]

The hypothesis: higher evidence → larger Δ.  When the model has seen an agent
visit a POI frequently during training, it can detect the behavioural absence
that signals a false belief.  For POIs the agent rarely or never visited, there
is no such signal and Δ should be near zero.

Evidence bins
-------------
  0  |  1–3  |  4–8  |  9–15  |  16+

Protocol
--------
  Evidence  : computed from train.jsonl (what the model saw during training)
    Δ         : computed from test.jsonl  (out-of-sample evaluation)

    Belief inference uses the standard eval protocol: same-agent context
    episodes are drawn from the evaluated split.

This separation ensures that evidence counts reflect genuine training exposure
while inference follows the model's intended evaluation setup.

Variable-tier POIs
------------------
Any POI that appears as a false belief in at least one belief config
(B0–B7, BH0–BH1) in config/belief_configs.json.

Output
------
  results/belief_evidence/results.json
    results/belief_evidence/evidence_sensitivity_paper.png

    results.json also contains diagnostics:
        - continuous log-evidence effect (with and without agent/POI fixed effects)
        - within-agent shuffled negative-control p-value

Usage
-----
    .venv/bin/python scripts/eval_belief_evidence.py
"""
from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader

from model.bdi_bottleneck import BDIBottleneckPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.meta_dataset import MetaEpisodeDataset
from simulation.episode import EpisodeRecord


# ── Evidence bins ──────────────────────────────────────────────────────

EVIDENCE_BINS = [
    (0,  0,  "0"),
    (1,  3,  "1–3"),
    (4,  8,  "4–8"),
    (9,  15, "9–15"),
    (16, 10_000, "16+"),
]

BIN_CENTERS = {
    "0": 0.0,
    "1–3": 2.0,
    "4–8": 6.0,
    "9–15": 12.0,
    "16+": 20.0,
}

_BBOX = {
    "lat_min": 32.867,
    "lat_max": 32.895,
    "lon_min": -117.253,
    "lon_max": -117.220,
}


def _bin_label(count: int) -> str | None:
    for lo, hi, label in EVIDENCE_BINS:
        if lo <= count <= hi:
            return label
    return None


def _norm(lat: float, lon: float) -> tuple[float, float]:
    lat_n = (lat - _BBOX["lat_min"]) / (_BBOX["lat_max"] - _BBOX["lat_min"])
    lon_n = (lon - _BBOX["lon_min"]) / (_BBOX["lon_max"] - _BBOX["lon_min"])
    return lat_n, lon_n


def _repeat_to(lst: list, n: int) -> list:
    return (lst * ((n // len(lst)) + 1))[:n]


# ── Load variable-tier POIs ────────────────────────────────────────────

def load_variable_tier_pois(belief_cfg_path: Path) -> set[str]:
    """
    Return the set of POI ids that appear as a false belief in any config.
    These are the only POIs for which agents can have false beliefs, and
    therefore the only POIs for which Δ is defined.
    """
    with open(belief_cfg_path) as f:
        cfg = json.load(f)

    variable_pois: set[str] = set()

    # Support both schema styles:
    # 1) {"configs": {"B0": {...}, ...}}
    # 2) {"training": {...}, "held_out": {...}}
    config_groups: list[dict] = []
    if isinstance(cfg.get("configs"), dict):
        config_groups.append(cfg["configs"])
    if isinstance(cfg.get("training"), dict):
        config_groups.append(cfg["training"])
    if isinstance(cfg.get("held_out"), dict):
        config_groups.append(cfg["held_out"])

    for group in config_groups:
        for config in group.values():
            for poi_id in config.get("false_beliefs", []):
                variable_pois.add(poi_id)

    return variable_pois


# ── Build evidence counts from train split ─────────────────────────────

def build_evidence_counts(
    train_records: list[EpisodeRecord],
) -> dict[tuple[str, str], int]:
    """
    evidence_counts[(agent_id, poi_id)] = number of train episodes where
    agent_id's goal_poi == poi_id.

    Pairs not present in the dict have count 0.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in train_records:
        counts[(r.agent_id, r.goal_poi)] += 1
    return counts


def build_evidence_counts_path_presence(
    train_records: list[EpisodeRecord],
    poi_to_node: dict[str, int],
) -> dict[tuple[str, str], int]:
    """
    evidence_counts[(agent_id, poi_id)] = number of train episodes where
    poi_id appears anywhere in the trajectory path.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in train_records:
        path_nodes = set(r.path)
        for poi_id, node_id in poi_to_node.items():
            if node_id in path_nodes:
                counts[(r.agent_id, poi_id)] += 1
    return counts


def _record_to_coords(r: EpisodeRecord, node_coords: dict[int, tuple[float, float]]) -> torch.Tensor:
    pts = [_norm(*node_coords.get(nid, (0.0, 0.0))) for nid in r.path]
    return torch.tensor(pts, dtype=torch.float32)


def _build_context_tensor(
    records_ctx: list[EpisodeRecord],
    node_coords: dict[int, tuple[float, float]],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [_record_to_coords(r, node_coords) for r in records_ctx]
    while len(tensors) < k:
        tensors.append(tensors[-1].clone() if tensors else torch.zeros(1, 2))
    tensors = tensors[:k]

    t_max = max(t.size(0) for t in tensors)
    coords = torch.zeros(k, t_max, 2)
    mask = torch.zeros(k, t_max, dtype=torch.long)
    for i, t in enumerate(tensors):
        t_len = t.size(0)
        coords[i, :t_len] = t
        mask[i, :t_len] = 1
    return coords, mask


def _build_agent_index(records: list[EpisodeRecord]) -> dict[str, list[int]]:
    idx_map: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        idx_map[r.agent_id].append(i)
    return idx_map


# ── Inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_belief_predictions(
    model:       BDIBottleneckPredictor,
    records:     list[EpisodeRecord],
    node_coords: dict,
    poi_to_idx:  dict[str, int],
    device,
    K:           int = 10,
    K_recent:    int = 5,
    batch_size:  int = 64,
) -> list[dict]:
    """
    Run the BDIBottleneck belief head on every episode.

    Contexts are drawn from the evaluated split (standard eval protocol).

    Returns a list of per-episode dicts:
      agent_id      : str
    false_beliefs : list[str]   POI ids the agent believes closed in this episode
      belief_probs  : list[float] P(believed open), length M
    """
    model.eval()

    ds = MetaEpisodeDataset(records, node_coords, poi_to_idx, K=K, K_recent=K_recent)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MetaEpisodeDataset.collate_fn,
        num_workers=0,
    )

    results: list[dict] = []
    sample_idx = 0
    for batch in loader:
        dc = batch["desire_coords"].to(device)
        dm = batch["desire_mask"].to(device)
        rc = batch["recent_coords"].to(device)
        rm = batch["recent_mask"].to(device)
        cc = batch["current_coords"].to(device)
        cm = batch["current_mask"].to(device)

        _, _, belief_logits = model(dc, dm, rc, rm, cc, cm)
        belief_probs = belief_logits.sigmoid().cpu()

        for i in range(belief_probs.size(0)):
            r = records[sample_idx]
            results.append({
                "agent_id": r.agent_id,
                "false_beliefs": r.false_beliefs,
                "belief_probs": belief_probs[i].tolist(),
            })
            sample_idx += 1

    return results


@torch.no_grad()
def collect_belief_predictions_excluding_same_config_context(
    model: BDIBottleneckPredictor,
    records: list[EpisodeRecord],
    node_coords: dict,
    poi_to_idx: dict[str, int],
    device,
    K: int = 10,
    K_recent: int = 5,
    batch_size: int = 64,
    rng_seed: int = 42,
) -> list[dict]:
    """
    Context ablation: build context from the evaluated split but exclude episodes
    with the same belief_config as the current episode when sampling context.
    If exclusion leaves no candidates, fall back to standard same-agent context.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)
    agent_idx = _build_agent_index(records)

    results: list[dict] = []

    for start in range(0, len(records), batch_size):
        batch_recs = records[start : start + batch_size]
        B = len(batch_recs)

        desire_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        recent_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        curr_list: list[torch.Tensor] = []

        for r in batch_recs:
            flat_idx = records.index(r)
            agent_idxs = agent_idx[r.agent_id]
            pos = agent_idxs.index(flat_idx)

            others = [i for i in agent_idxs if i != flat_idx]
            others_excl_same_cfg = [i for i in others if records[i].belief_config != r.belief_config]
            desire_pool = others_excl_same_cfg if len(others_excl_same_cfg) > 0 else others

            if len(desire_pool) == 0:
                d_src = [flat_idx] * K
            elif len(desire_pool) < K:
                d_src = rng.choice(desire_pool, size=K, replace=True).tolist()
            else:
                d_src = rng.choice(desire_pool, size=K, replace=False).tolist()

            preceding = agent_idxs[:pos]
            preceding_excl_same_cfg = [i for i in preceding if records[i].belief_config != r.belief_config]

            if len(preceding_excl_same_cfg) >= K_recent:
                r_src = preceding_excl_same_cfg[-K_recent:]
            elif len(preceding_excl_same_cfg) > 0:
                r_src = _repeat_to(preceding_excl_same_cfg, K_recent)
            elif len(preceding) >= K_recent:
                r_src = preceding[-K_recent:]
            elif len(preceding) > 0:
                r_src = _repeat_to(preceding, K_recent)
            else:
                fallback = [i for i in others_excl_same_cfg] if len(others_excl_same_cfg) > 0 else [i for i in others]
                if len(fallback) == 0:
                    fallback = [flat_idx]
                r_src = _repeat_to(fallback, K_recent)

            desire_ctx_list.append(_build_context_tensor([records[i] for i in d_src], node_coords, K))
            recent_ctx_list.append(_build_context_tensor([records[i] for i in r_src], node_coords, K_recent))
            curr_list.append(_record_to_coords(r, node_coords))

        def stack_context(ctx_list):
            K_ = ctx_list[0][0].size(0)
            T_ = max(c.size(1) for c, _ in ctx_list)
            B_ = len(ctx_list)
            co = torch.zeros(B_, K_, T_, 2)
            ma = torch.zeros(B_, K_, T_, dtype=torch.long)
            for i, (c, m) in enumerate(ctx_list):
                t = c.size(1)
                co[i, :, :t, :] = c
                ma[i, :, :t] = m
            return co, ma

        dc, dm = stack_context(desire_ctx_list)
        rc, rm = stack_context(recent_ctx_list)

        T_c = max(c.size(0) for c in curr_list)
        cc = torch.zeros(B, T_c, 2)
        cm = torch.zeros(B, T_c, dtype=torch.long)
        for i, c in enumerate(curr_list):
            cc[i, : c.size(0)] = c
            cm[i, : c.size(0)] = 1

        _, _, belief_logits = model(
            dc.to(device), dm.to(device),
            rc.to(device), rm.to(device),
            cc.to(device), cm.to(device),
        )
        belief_probs = belief_logits.sigmoid().cpu()

        for i in range(B):
            r = batch_recs[i]
            results.append({
                "agent_id": r.agent_id,
                "false_beliefs": r.false_beliefs,
                "belief_probs": belief_probs[i].tolist(),
            })

    return results


# ── Compute per-(agent, poi) Δ ─────────────────────────────────────────

def compute_delta_pairs(
    episode_results: list[dict],
    variable_pois:   set[str],
    poi_to_idx:      dict[str, int],
    evidence_counts: dict[tuple[str, str], int],
) -> list[dict]:
    """
    For each (agent_id, poi_id) pair where poi_id is a Variable-tier POI,
    compute:
      evidence : int   training visit count for this agent–POI pair
    delta    : float mean P̂(closed | b_j=0) − mean P̂(closed | b_j=1)
      n_closed : int   number of test episodes where this POI was in false_beliefs
      n_open   : int   number of test episodes where it was not

    Only pairs that have observations in BOTH conditions are returned.
    """
    # Accumulate P_closed observations per (agent, poi, condition)
    # condition ∈ {"closed", "open"}
    obs: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"closed": [], "open": []}
    )

    for r in episode_results:
        agent_id   = r["agent_id"]
        probs      = r["belief_probs"]
        fb_set     = set(r["false_beliefs"])

        for poi_id in sorted(variable_pois):  # sorted: set order varies per process
            if poi_id not in poi_to_idx:
                continue
            idx = poi_to_idx[poi_id]
            p_closed = 1.0 - probs[idx]   # P(believed closed)
            condition = "closed" if poi_id in fb_set else "open"
            obs[(agent_id, poi_id)][condition].append(p_closed)

    pairs: list[dict] = []
    for (agent_id, poi_id), d in obs.items():
        if not d["closed"] or not d["open"]:
            continue   # need both conditions
        evidence = evidence_counts.get((agent_id, poi_id), 0)
        mean_closed = float(np.mean(d["closed"]))
        mean_open   = float(np.mean(d["open"]))
        pairs.append({
            "agent_id":   agent_id,
            "poi_id":     poi_id,
            "evidence":   evidence,
            "mean_p_closed_truly_closed": mean_closed,
            "mean_p_closed_truly_open":   mean_open,
            "delta":      mean_closed - mean_open,
            "n_closed":   len(d["closed"]),
            "n_open":     len(d["open"]),
        })

    return pairs


# ── Bin and aggregate ──────────────────────────────────────────────────

def aggregate_by_evidence_bin(pairs: list[dict]) -> list[dict]:
    """
    Group (agent, POI) pairs by evidence bin, compute mean Δ and ±1 SE
    per bin.

    Returns a list of dicts (one per bin, in EVIDENCE_BINS order):
      label     : str
      mean_delta: float
      se_delta  : float
      n_pairs   : int
      bin_lo    : int
      bin_hi    : int
    """
    binned: dict[str, list[float]] = defaultdict(list)

    for p in pairs:
        label = _bin_label(p["evidence"])
        if label is not None:
            binned[label].append(p["delta"])

    rows: list[dict] = []
    for lo, hi, label in EVIDENCE_BINS:
        deltas = binned.get(label, [])
        n = len(deltas)
        if n == 0:
            mean_d = 0.0
            se_d   = 0.0
        else:
            arr    = np.array(deltas)
            mean_d = float(arr.mean())
            se_d   = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append({
            "label":      label,
            "mean_delta": mean_d,
            "se_delta":   se_d,
            "n_pairs":    n,
            "bin_lo":     lo,
            "bin_hi":     hi,
        })
    return rows


def _slope_from_bins(rows: list[dict]) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        if r["n_pairs"] <= 0:
            continue
        xs.append(BIN_CENTERS[r["label"]])
        ys.append(r["mean_delta"])
    if len(xs) < 2:
        return float("nan")
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def _beta_logevidence(y: np.ndarray, x: np.ndarray, agent_ids: list[str], poi_ids: list[str], with_fe: bool) -> float:
    cols = [np.ones_like(x), x]
    if with_fe:
        unique_agents = sorted(set(agent_ids))
        unique_pois = sorted(set(poi_ids))
        for a in unique_agents[1:]:
            cols.append((np.array(agent_ids) == a).astype(float))
        for p in unique_pois[1:]:
            cols.append((np.array(poi_ids) == p).astype(float))
    x_mat = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    return float(coef[1])


def compute_diagnostics(
    pairs: list[dict],
    observed_bins: list[dict],
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    if not pairs:
        return {
            "n_pairs": 0,
            "continuous": {},
            "negative_control": {},
        }

    rng = np.random.default_rng(seed)
    evidence = np.array([p["evidence"] for p in pairs], dtype=float)
    x = np.log1p(evidence)
    y = np.array([p["delta"] for p in pairs], dtype=float)
    agent_ids = [p["agent_id"] for p in pairs]
    poi_ids = [p["poi_id"] for p in pairs]

    beta_no_fe = _beta_logevidence(y, x, agent_ids, poi_ids, with_fe=False)
    beta_fe = _beta_logevidence(y, x, agent_ids, poi_ids, with_fe=True)

    perm_betas = []
    for _ in range(n_perm):
        x_perm = rng.permutation(x)
        perm_betas.append(_beta_logevidence(y, x_perm, agent_ids, poi_ids, with_fe=True))
    perm_betas_arr = np.array(perm_betas)
    p_beta_fe = float((np.sum(perm_betas_arr >= beta_fe) + 1) / (n_perm + 1))

    observed_slope = _slope_from_bins(observed_bins)

    idx_by_agent: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        idx_by_agent[p["agent_id"]].append(i)

    shuffled_slopes = []
    for _ in range(n_perm):
        shuffled_evidence = evidence.copy()
        for idxs in idx_by_agent.values():
            shuffled_evidence[idxs] = rng.permutation(shuffled_evidence[idxs])

        shuffled_pairs = []
        for i, p in enumerate(pairs):
            q = dict(p)
            q["evidence"] = int(shuffled_evidence[i])
            shuffled_pairs.append(q)
        bins_perm = aggregate_by_evidence_bin(shuffled_pairs)
        shuffled_slopes.append(_slope_from_bins(bins_perm))

    shuffled_slopes_arr = np.array([s for s in shuffled_slopes if np.isfinite(s)], dtype=float)
    if np.isfinite(observed_slope) and shuffled_slopes_arr.size > 0:
        p_slope = float((np.sum(shuffled_slopes_arr >= observed_slope) + 1) / (shuffled_slopes_arr.size + 1))
    else:
        p_slope = float("nan")

    return {
        "n_pairs": len(pairs),
        "continuous": {
            "beta_log1p_evidence_no_fe": beta_no_fe,
            "beta_log1p_evidence_with_agent_poi_fe": beta_fe,
            "perm_p_one_sided_beta_fe_gt0": p_beta_fe,
            "n_permutations": n_perm,
        },
        "negative_control": {
            "observed_bin_slope": observed_slope,
            "mean_shuffled_bin_slope": float(np.nanmean(shuffled_slopes_arr)) if shuffled_slopes_arr.size > 0 else float("nan"),
            "perm_p_one_sided_observed_gt_shuffled": p_slope,
            "n_permutations": int(shuffled_slopes_arr.size),
            "shuffle_scheme": "shuffle evidence within agent across POIs",
        },
    }


# ── Plot ───────────────────────────────────────────────────────────────

def plot_evidence_sensitivity(
    split_bins: dict[str, list[dict]],
    out_path:   Path,
    paper:      bool = True,
) -> None:
    """
    Bar chart of mean Δ (±1 SE) per evidence bin, one subplot per split.
    """
    splits   = list(split_bins.keys())
    n_splits = len(splits)

    if n_splits == 0:
        print("  WARNING: No split data available for plotting.")
        return

    fig, axes = plt.subplots(
        1, n_splits, figsize=(5.5 * n_splits, 4.5), squeeze=False
    )

    bar_color   = "#2196F3"
    edge_color  = "#1565C0"
    error_color = "#0D47A1"

    # Use a shared y-scale across subplots so split comparisons are visually fair.
    y_lows: list[float] = []
    y_highs: list[float] = []
    for split_name in splits:
        rows = split_bins[split_name]
        means = [r["mean_delta"] * 100 for r in rows]
        ses = [r["se_delta"] * 100 for r in rows]
        if means:
            y_lows.append(min(m - s for m, s in zip(means, ses)))
            y_highs.append(max(m + s for m, s in zip(means, ses)))

    y_min = min(y_lows) if y_lows else -1.0
    y_max = max(y_highs) if y_highs else 1.0
    y_pad = max(1.0, 0.12 * (y_max - y_min if y_max > y_min else 1.0))
    y_min -= y_pad
    y_max += y_pad

    for i, (ax, split_name) in enumerate(zip(axes[0], splits)):
        rows = split_bins[split_name]

        labels     = [r["label"]      for r in rows]
        means      = [r["mean_delta"] * 100 for r in rows]
        ses        = [r["se_delta"]   * 100 for r in rows]
        n_pairs    = [r["n_pairs"]    for r in rows]

        # Do not draw bars for empty bins; this avoids implying a true 0 effect.
        means_plot = [m if n > 0 else np.nan for m, n in zip(means, n_pairs)]
        ses_plot   = [s if n > 0 else np.nan for s, n in zip(ses, n_pairs)]

        x = np.arange(len(labels))

        bars = ax.bar(
            x, means_plot,
            color=bar_color, edgecolor=edge_color, linewidth=0.8,
            width=0.6, zorder=3,
        )
        ax.errorbar(
            x, means_plot, yerr=ses_plot,
            fmt="none", color=error_color,
            capsize=5, linewidth=1.5, zorder=4,
        )

        # Count labels near each bar end.
        for xi, mean_val, se_val, n in zip(x, means, ses, n_pairs):
            if n == 0:
                continue
            if mean_val >= 0:
                y_txt = mean_val + se_val + 0.4
                va = "bottom"
            else:
                y_txt = mean_val - se_val - 0.4
                va = "top"
            ax.text(xi, y_txt, f"pairs={n}", ha="center", va=va,
                    fontsize=9, color="#555555")

        ax.axhline(0, color="gray", linewidth=1.0, linestyle=":", alpha=0.7, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel("Evidence (training visits to POI)", fontsize=11)
        if i == 0:
            ax.set_ylabel(r"$\Delta$", fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(y_min, y_max)

    dpi = 300
    fig.tight_layout(pad=2.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Belief-head sensitivity as a function of agent-specific evidence. "
            "Evidence = training visit count per (agent, POI); "
            "Δ = P̂(closed|b_j=0) − P̂(closed|b_j=1) on the test split."
        )
    )
    p.add_argument("--bdi-ckpt",    default="outputs/bdi_bottleneck/best.pt", metavar="PT")
    p.add_argument("--splits-dir",  default="data/splits",                    metavar="DIR")
    p.add_argument("--train-split", default="train",                          metavar="SPLIT",
                   help="Split used to compute evidence counts (default: train)")
    p.add_argument("--eval-split",  default="test",                           metavar="SPLIT",
                   help="Split used for Δ evaluation (default: test)")
    p.add_argument("--K",           type=int, default=10,  metavar="K")
    p.add_argument("--K-recent",    type=int, default=5,   metavar="K", dest="K_recent")
    p.add_argument("--seed",        type=int, default=0,   metavar="N",
                   help="Random seed for diagnostic permutations")
    p.add_argument("--batch",       type=int, default=64,  metavar="N")
    p.add_argument("--n-perm",      type=int, default=2000, metavar="N",
                   help="Number of permutations for diagnostic tests")
    p.add_argument("--no-cuda",     action="store_true")
    # Backward-compatibility no-op: output is always paper-style now.
    p.add_argument("--paper", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def _load_records(path: Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_json(line))
    return records


def main() -> None:
    args       = parse_args()
    splits_dir = ROOT / args.splits_dir
    out_dir    = ROOT / "results" / "belief_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_split != "test":
        print(
            f"ERROR: This script is configured to evaluate Δ on the test split only. "
            f"Got --eval-split {args.eval_split!r}."
        )
        sys.exit(1)

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
    poi_to_node = {
        p["id"]: int(p["node_id"])
        for cd in poi_cfg["categories"].values()
        for p in cd["pois"]
    }

    # ── Variable-tier POIs ─────────────────────────────────────────────
    variable_pois = load_variable_tier_pois(ROOT / "config" / "belief_configs.json")
    if not variable_pois:
        print("ERROR: No variable-tier POIs found in config/belief_configs.json.")
        print("       Expected false_beliefs under either 'configs' or 'training'/'held_out'.")
        sys.exit(1)
    print(f"Variable-tier POIs: {len(variable_pois)}")
    print(f"  {sorted(variable_pois)}\n")

    # ── Graph ─────────────────────────────────────────────────────────
    import osmnx as ox
    print("Loading graph …")
    G           = ox.load_graphml(ROOT / "data" / "ucsd_walk.graphml")
    node_coords = node_coords_from_graph(G)
    print(f"  {len(node_coords):,} nodes\n")

    # ── Model ─────────────────────────────────────────────────────────
    ckpt_path = ROOT / args.bdi_ckpt
    ckpt      = torch.load(ckpt_path, map_location=device)
    state     = ckpt["model_state"]
    d_model   = state["coord_proj.weight"].shape[0]
    num_pois  = state["belief_head.weight"].shape[0]

    model = BDIBottleneckPredictor(num_pois=num_pois, d_model=d_model).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"BDIBottleneckPredictor loaded  ({sum(p.numel() for p in model.parameters()):,} params)\n")

    # ── Evidence counts (from train split) ────────────────────────────
    train_path = splits_dir / f"{args.train_split}.jsonl"
    if not train_path.exists():
        print(f"ERROR: train split not found at {train_path}")
        sys.exit(1)

    print(f"Loading train split for evidence counts: {train_path}")
    train_records = _load_records(train_path)
    evidence_counts_goal = build_evidence_counts(train_records)
    evidence_counts_path = build_evidence_counts_path_presence(train_records, poi_to_node)
    print(f"  {len(train_records)} train episodes")
    print(f"  goal-evidence: {len(evidence_counts_goal)} (agent, POI) pairs with evidence > 0")
    print(f"  path-evidence: {len(evidence_counts_path)} (agent, POI) pairs with evidence > 0\n")

    # ── Δ evaluation on test split only ───────────────────────────────
    split_pairs: dict[str, list[dict]] = {}
    split_bins:  dict[str, list[dict]] = {}
    split_name = args.eval_split
    path = splits_dir / f"{split_name}.jsonl"
    if not path.exists():
        print(f"ERROR: evaluation split not found at {path}")
        sys.exit(1)

    print(f"  {split_name} — loading episodes …")
    records = _load_records(path)
    print(f"    {len(records)} episodes — running belief inference …")

    episode_results_standard = collect_belief_predictions(
        model,
        records,
        node_coords=node_coords,
        poi_to_idx=poi_to_idx,
        device=device,
        K=args.K,
        K_recent=args.K_recent,
        batch_size=args.batch,
    )

    episode_results_context_ablation = collect_belief_predictions_excluding_same_config_context(
        model,
        records,
        node_coords=node_coords,
        poi_to_idx=poi_to_idx,
        device=device,
        K=args.K,
        K_recent=args.K_recent,
        batch_size=args.batch,
        rng_seed=args.seed,
    )

    analyses = [
        {
            "key": "original",
            "description": "Goal evidence from train; standard test-split context",
            "episode_results": episode_results_standard,
            "evidence_counts": evidence_counts_goal,
            "plot_stem": "evidence_sensitivity",
            "evidence_definition": "count of train episodes where goal_poi == poi_id",
            "context_definition": "standard eval context from evaluated split",
        },
        {
            "key": "ablation_context_excluding_same_config",
            "description": "Goal evidence from train; context excludes same belief_config episodes when possible",
            "episode_results": episode_results_context_ablation,
            "evidence_counts": evidence_counts_goal,
            "plot_stem": "evidence_sensitivity_ablation_context_excluding_same_config",
            "evidence_definition": "count of train episodes where goal_poi == poi_id",
            "context_definition": "evaluated-split context excluding same belief_config where possible",
        },
        {
            "key": "ablation_path_presence_evidence",
            "description": "Path-presence evidence from train; standard test-split context",
            "episode_results": episode_results_standard,
            "evidence_counts": evidence_counts_path,
            "plot_stem": "evidence_sensitivity_ablation_path_presence_evidence",
            "evidence_definition": "count of train episodes where poi node appears anywhere in path",
            "context_definition": "standard eval context from evaluated split",
        },
    ]

    analysis_outputs: dict[str, dict] = {}

    for a in analyses:
        print(f"  Analysis: {a['key']}")
        print(f"    {a['description']}")

        pairs = compute_delta_pairs(
            a["episode_results"],
            variable_pois,
            poi_to_idx,
            a["evidence_counts"],
        )
        bins = aggregate_by_evidence_bin(pairs)

        print(f"    {len(pairs)} (agent, POI) pairs with both conditions")
        print("    Evidence-bin summary:")
        for row in bins:
            bar = "█" * int(max(0, row["mean_delta"] * 200))
            print(
                f"      {row['label']:>5}  "
                f"Δ={row['mean_delta']*100:+6.2f}% ±{row['se_delta']*100:.2f}%  "
                f"n={row['n_pairs']:4d}  {bar}"
            )

        diagnostics = compute_diagnostics(
            pairs,
            observed_bins=bins,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        cont = diagnostics["continuous"]
        neg = diagnostics["negative_control"]
        print("    Continuous evidence-effect diagnostics:")
        print(f"      beta(log1p evidence), no FE       = {cont.get('beta_log1p_evidence_no_fe', float('nan')):+.6f}")
        print(f"      beta(log1p evidence), agent+POI FE = {cont.get('beta_log1p_evidence_with_agent_poi_fe', float('nan')):+.6f}")
        print(f"      perm p (H1: beta>0, FE model)      = {cont.get('perm_p_one_sided_beta_fe_gt0', float('nan')):.4f}")
        print("    Negative-control (within-agent shuffled evidence):")
        print(f"      observed bin-slope                 = {neg.get('observed_bin_slope', float('nan')):+.6f}")
        print(f"      mean shuffled bin-slope            = {neg.get('mean_shuffled_bin_slope', float('nan')):+.6f}")
        print(f"      perm p (H1: observed > shuffled)   = {neg.get('perm_p_one_sided_observed_gt_shuffled', float('nan')):.4f}")
        print()

        analysis_outputs[a["key"]] = {
            "description": a["description"],
            "evidence_definition": a["evidence_definition"],
            "context_definition": a["context_definition"],
            "pairs": pairs,
            "bins": bins,
            "diagnostics": diagnostics,
        }

        suffix = "_paper"
        plot_evidence_sensitivity(
            {split_name: bins},
            out_dir / f"{a['plot_stem']}{suffix}.png",
            paper=True,
        )

    original = analysis_outputs["original"]
    split_pairs[split_name] = original["pairs"]
    split_bins[split_name] = original["bins"]
    diagnostics = original["diagnostics"]

    if not split_bins:
        print("ERROR: No valid evaluation splits were processed; cannot plot or save results.")
        sys.exit(1)

    # ── Save JSON ──────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "variable_pois":   sorted(variable_pois),
                "evidence_bins":   [
                    {"lo": lo, "hi": hi, "label": lbl}
                    for lo, hi, lbl in EVIDENCE_BINS
                ],
                "splits": {
                    split_name: {
                        "pairs": split_pairs[split_name],
                        "bins":  split_bins[split_name],
                        "diagnostics": diagnostics,
                    }
                    for split_name in split_pairs
                },
                "analyses": analysis_outputs,
            },
            f, indent=2,
        )
    print(f"  Results → {results_path}")
    print(f"  Plots   → {out_dir}/")


if __name__ == "__main__":
    main()
