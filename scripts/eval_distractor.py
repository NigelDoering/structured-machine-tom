"""
scripts/eval_distractor.py

Evaluate GoalPredictor, ContextGoalPredictor, and BDIBottleneck on the
distractor episode subset.  All models are evaluated at the "distractor moment"
— the path step where the agent is closest to the low-preference POI.  This
is the hardest inference point: spatially the distractor most resembles the
destination.

Metrics (per model, per split)
-------------------------------
  mean_p_true      : mean softmax probability assigned to the true goal POI
  mean_p_distract  : mean softmax probability assigned to the distractor POI
  likelihood_ratio : mean P(true goal) / P(distractor)   [higher = better]
  pct_correct      : % episodes where P(true goal) > P(distractor)  [headline]

Output
------
  results/distractor_eval/results.json
  results/distractor_eval/comparison.png

Usage
-----
  .venv/bin/python scripts/eval_distractor.py
  .venv/bin/python scripts/eval_distractor.py \\
      --goal-ckpt  outputs/goal_predictor/best.pt \\
      --bnk-ckpt   outputs/bdi_bottleneck/best.pt
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
import torch.nn.functional as F

from model.bdi_bottleneck import BDIBottleneckPredictor
from model.context_goal_predictor import ContextGoalPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.goal_predictor import GoalPredictor
from model.meta_dataset import _norm
from simulation.episode import EpisodeRecord
from scripts.train_bdi_predictor import (
    _build_agent_index,
    _build_context_tensor,
)


# ── Coordinate helpers ─────────────────────────────────────────────────

def _record_to_coords(r: EpisodeRecord, node_coords: dict) -> torch.Tensor:
    pts = [_norm(*node_coords.get(nid, (0.0, 0.0))) for nid in r.path]
    return torch.tensor(pts, dtype=torch.float32)


def _truncate_coords(r: EpisodeRecord, node_coords: dict, up_to_idx: int) -> torch.Tensor:
    """Path coordinates truncated to up_to_idx + 1 steps (inclusive)."""
    end = min(up_to_idx + 1, len(r.path))
    end = max(end, 1)
    pts = [_norm(*node_coords.get(nid, (0.0, 0.0))) for nid in r.path[:end]]
    return torch.tensor(pts, dtype=torch.float32)


def _pad_sequence(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (T_i, 2) tensors to (B, T_max, 2) with mask."""
    B     = len(tensors)
    T_max = max(t.size(0) for t in tensors)
    coords = torch.zeros(B, T_max, 2)
    mask   = torch.zeros(B, T_max, dtype=torch.long)
    for i, t in enumerate(tensors):
        T = t.size(0)
        coords[i, :T] = t
        mask[i,   :T] = 1
    return coords, mask


def _repeat_to(lst: list, n: int) -> list:
    return (lst * ((n // len(lst)) + 1))[:n]


# ── Per-episode inference ──────────────────────────────────────────────

@torch.no_grad()
def run_goal_predictor(
    model:       GoalPredictor,
    records:     list[EpisodeRecord],
    meta:        list[dict],
    node_coords: dict,
    device,
    batch_size:  int = 64,
) -> list[dict[str, float]]:
    """
    Run GoalPredictor at the distractor moment for each episode.
    Returns list of {p_true, p_distract} dicts in the same order as records.
    """
    model.eval()
    results: list[dict[str, float]] = [{}] * len(records)

    for start in range(0, len(records), batch_size):
        batch_r = records[start : start + batch_size]
        batch_m = meta[start : start + batch_size]

        curr_tensors = [
            _truncate_coords(r, node_coords, m["closest_path_idx"])
            for r, m in zip(batch_r, batch_m)
        ]
        coords, mask = _pad_sequence(curr_tensors)
        coords, mask = coords.to(device), mask.to(device)

        logits = model(coords, mask)            # (B, M)
        probs  = F.softmax(logits, dim=-1).cpu()

        for i, (r, m) in enumerate(zip(batch_r, batch_m)):
            results[start + i] = {
                "p_true":     probs[i, poi_to_idx[r.goal_poi]].item(),
                "p_distract": probs[i, poi_to_idx[m["distractor_poi"]]].item(),
            }

    return results


@torch.no_grad()
def run_bdi_predictor(
    model:       BDIPredictor,
    records:     list[EpisodeRecord],
    meta:        list[dict],
    node_coords: dict,
    poi_to_idx:  dict[str, int],
    device,
    K:           int = 10,
    K_recent:    int = 5,
    batch_size:  int = 64,
    rng_seed:    int = 42,
) -> list[dict[str, float]]:
    """
    Run BDIPredictor at the distractor moment for each episode.
    Constructs desire and recent contexts exactly as during training.
    Returns list of {p_true, p_distract} dicts.
    """
    model.eval()
    rng       = np.random.default_rng(rng_seed)
    agent_idx = _build_agent_index(records)
    results: list[dict[str, float]] = [{}] * len(records)

    for start in range(0, len(records), batch_size):
        batch_r = records[start : start + batch_size]
        batch_m = meta[start : start + batch_size]
        B       = len(batch_r)

        desire_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        recent_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        curr_list:       list[torch.Tensor]                      = []

        for r in batch_r:
            flat_idx   = records.index(r)
            agent_idxs = agent_idx[r.agent_id]
            pos        = agent_idxs.index(flat_idx)
            others     = [i for i in agent_idxs if i != flat_idx]

            # Desire context: K random past trajectories
            if len(others) == 0:
                d_src = [flat_idx] * K
            elif len(others) < K:
                d_src = rng.choice(others, size=K, replace=True).tolist()
            else:
                d_src = rng.choice(others, size=K, replace=False).tolist()

            # Recent context: K_recent most-recent preceding
            preceding = agent_idxs[:pos]
            if len(preceding) == 0:
                fallback = agent_idxs[pos + 1:] or [flat_idx]
                r_src = _repeat_to(fallback, K_recent)
            elif len(preceding) < K_recent:
                r_src = _repeat_to(preceding, K_recent)
            else:
                r_src = preceding[-K_recent:]

            desire_ctx_list.append(
                _build_context_tensor([records[i] for i in d_src], node_coords, K)
            )
            recent_ctx_list.append(
                _build_context_tensor([records[i] for i in r_src], node_coords, K_recent)
            )

        # Current trajectory truncated to distractor moment
        for r, m in zip(batch_r, batch_m):
            curr_list.append(_truncate_coords(r, node_coords, m["closest_path_idx"]))

        def stack_ctx(ctx_list):
            K_  = ctx_list[0][0].size(0)
            T_  = max(c.size(1) for c, _ in ctx_list)
            co  = torch.zeros(B, K_, T_, 2)
            ma  = torch.zeros(B, K_, T_, dtype=torch.long)
            for i, (c, m_) in enumerate(ctx_list):
                t = c.size(1)
                co[i, :, :t, :] = c
                ma[i, :, :t]    = m_
            return co, ma

        dc, dm = stack_ctx(desire_ctx_list)
        rc, rm = stack_ctx(recent_ctx_list)
        cc, cm = _pad_sequence(curr_list)

        goal_logits, _, _ = model(
            dc.to(device), dm.to(device),
            rc.to(device), rm.to(device),
            cc.to(device), cm.to(device),
        )
        probs = F.softmax(goal_logits, dim=-1).cpu()

        for i, (r, m) in enumerate(zip(batch_r, batch_m)):
            results[start + i] = {
                "p_true":     probs[i, poi_to_idx[r.goal_poi]].item(),
                "p_distract": probs[i, poi_to_idx[m["distractor_poi"]]].item(),
            }

    return results


@torch.no_grad()
def run_ctx_predictor(
    model:       ContextGoalPredictor,
    records:     list[EpisodeRecord],
    meta:        list[dict],
    node_coords: dict,
    poi_to_idx:  dict[str, int],
    device,
    K:           int = 10,
    K_recent:    int = 5,
    batch_size:  int = 64,
    rng_seed:    int = 42,
) -> list[dict[str, float]]:
    """
    Run ContextGoalPredictor at the distractor moment for each episode.
    Same context construction as run_bdi_predictor; returns a single goal
    tensor (no auxiliary heads).
    Returns list of {p_true, p_distract} dicts.
    """
    model.eval()
    rng       = np.random.default_rng(rng_seed)
    agent_idx = _build_agent_index(records)
    results: list[dict[str, float]] = [{}] * len(records)

    for start in range(0, len(records), batch_size):
        batch_r = records[start : start + batch_size]
        batch_m = meta[start : start + batch_size]
        B       = len(batch_r)

        desire_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        recent_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        curr_list:       list[torch.Tensor]                      = []

        for r in batch_r:
            flat_idx   = records.index(r)
            agent_idxs = agent_idx[r.agent_id]
            pos        = agent_idxs.index(flat_idx)
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
                r_src = _repeat_to(fallback, K_recent)
            elif len(preceding) < K_recent:
                r_src = _repeat_to(preceding, K_recent)
            else:
                r_src = preceding[-K_recent:]

            desire_ctx_list.append(
                _build_context_tensor([records[i] for i in d_src], node_coords, K)
            )
            recent_ctx_list.append(
                _build_context_tensor([records[i] for i in r_src], node_coords, K_recent)
            )

        for r, m in zip(batch_r, batch_m):
            curr_list.append(_truncate_coords(r, node_coords, m["closest_path_idx"]))

        def stack_ctx(ctx_list):
            K_  = ctx_list[0][0].size(0)
            T_  = max(c.size(1) for c, _ in ctx_list)
            co  = torch.zeros(B, K_, T_, 2)
            ma  = torch.zeros(B, K_, T_, dtype=torch.long)
            for i, (c, m_) in enumerate(ctx_list):
                t = c.size(1)
                co[i, :, :t, :] = c
                ma[i, :, :t]    = m_
            return co, ma

        dc, dm = stack_ctx(desire_ctx_list)
        rc, rm = stack_ctx(recent_ctx_list)
        cc, cm = _pad_sequence(curr_list)

        goal_logits = model(
            dc.to(device), dm.to(device),
            rc.to(device), rm.to(device),
            cc.to(device), cm.to(device),
        )
        probs = F.softmax(goal_logits, dim=-1).cpu()

        for i, (r, m) in enumerate(zip(batch_r, batch_m)):
            results[start + i] = {
                "p_true":     probs[i, poi_to_idx[r.goal_poi]].item(),
                "p_distract": probs[i, poi_to_idx[m["distractor_poi"]]].item(),
            }

    return results


# ── Metrics ────────────────────────────────────────────────────────────

def compute_metrics(preds: list[dict[str, float]]) -> dict[str, float]:
    p_true     = np.array([p["p_true"]     for p in preds])
    p_distract = np.array([p["p_distract"] for p in preds])
    ratio      = p_true / np.maximum(p_distract, 1e-9)
    return {
        "n":               len(preds),
        "mean_p_true":     float(p_true.mean()),
        "mean_p_distract": float(p_distract.mean()),
        "mean_ratio":      float(ratio.mean()),
        "pct_correct":     float((p_true > p_distract).mean() * 100),
    }


# ── Plotting ───────────────────────────────────────────────────────────

_MODEL_STYLE = {
    "GoalPredictor":        {"color": "#2196F3", "label": "GoalPredictor (baseline)"},
    "ContextGoalPredictor": {"color": "#FF9800", "label": "ContextGoalPredictor"},
    "BDIBottleneck":        {"color": "#4CAF50", "label": "BDIBottleneck (Ours)"},
}

_SPLIT_TITLE = {
    "test":           "Test Set",
    "test_new_agent": "Test Set (New Agent)",
}


def _draw_bar_panel(
    ax,
    config_metrics: dict[str, dict[str, dict]],
    panel_title:    str,
    show_legend:    bool,
    show_xlabel:    bool,
    paper:          bool = False,
    config_counts:  dict[str, int] | None = None,
) -> None:
    """Draw one bar panel onto ax."""
    all_keys = sorted(
        k for k in next(iter(config_metrics.values())).keys()
        if k != "Overall"
    ) + ["Overall"]
    models  = list(config_metrics.keys())
    x       = np.arange(len(all_keys))
    width   = 0.25
    offsets = np.linspace(-(len(models) - 1) * width / 2,
                            (len(models) - 1) * width / 2,
                            len(models))
    label_groups: list[list[tuple[float, float, str]]] = [[] for _ in all_keys]

    for model_i, (model_name, offset) in enumerate(zip(models, offsets)):
        style  = _MODEL_STYLE.get(model_name, {"color": "gray", "label": model_name})
        values = [config_metrics[model_name].get(k, {}).get("pct_correct", 0.0)
                  for k in all_keys]
        bars   = ax.bar(x + offset, values, width,
                        label=style["label"], color=style["color"], alpha=0.85)
        for cfg_i, (bar, v) in enumerate(zip(bars, values)):
            cx = bar.get_x() + bar.get_width() / 2
            # Keep labels from adjacent bars readable: stagger within each config group.
            x_nudge = (model_i - (len(models) - 1) / 2.0) * (width * 0.10)
            label_groups[cfg_i].append((cx + x_nudge, float(v), f"{v:.0f}%"))

    max_label_y = 0.0
    for group in label_groups:
        # Place lower values first; if too close, stack upward.
        group_sorted = sorted(group, key=lambda t: t[1])
        prev_y = -1e9
        for cx, v, txt in group_sorted:
            y = v + 0.8
            if y - prev_y < 2.0:
                y = prev_y + 2.0
            ax.text(cx, y, txt, ha="center", va="bottom", fontsize=9)
            prev_y = y
            if y > max_label_y:
                max_label_y = y

    ax.axvline(len(all_keys) - 1.5, color="lightgray", linewidth=1.5, linestyle="--")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.6, label="Chance (50%)")
    ax.set_xticks(x)

    # Build x-tick labels with per-config episode percentages if counts provided
    if config_counts is not None:
        total = config_counts.get("Overall", sum(
            v for k, v in config_counts.items() if k != "Overall"
        ))
        tick_labels = []
        for k in all_keys:
            if k == "Overall":
                tick_labels.append(f"Overall\n(N={total})")
            else:
                n   = config_counts.get(k, 0)
                pct = 100 * n / total if total > 0 else 0.0
                tick_labels.append(f"{k}\n({pct:.1f}%)")
    else:
        tick_labels = all_keys

    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("% Correct Ranking", fontsize=11)
    ax.set_ylim(0, max(115, max_label_y + 3.0))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_xlabel:
        ax.set_xlabel("Belief Configuration", fontsize=11)
    if not paper:
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
    else:
        # In paper mode annotate the split name as a left-aligned text inside the panel
        ax.text(0.01, 0.97, panel_title, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")
    if show_legend:
        ax.legend(
            fontsize=9.5,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.14),
            ncol=4,
            frameon=False,
            borderaxespad=0.0,
        )


def plot_comparison(
    split_config_metrics: dict[str, dict[str, dict[str, dict]]],  # {split: {model: {config: metrics}}}
    out_path:             Path,
    paper:                bool = False,
) -> None:
    """
    Stacked two-panel figure: top row = Test, bottom row = Test (New Agent).
    Each panel shows % correct ranking per belief config (B0–B7) + Overall.

    paper=True  : split labels inset, no matplotlib titles, 300 DPI.
    paper=False : titles as panel headings, 150 DPI.
    """
    splits = [s for s in ("test", "test_new_agent") if s in split_config_metrics]
    fig, axes = plt.subplots(len(splits), 1, figsize=(11, 5 * len(splits)))
    if len(splits) == 1:
        axes = [axes]

    for ax, split_name in zip(axes, splits):
        is_last  = (split_name == splits[-1])
        is_first = (split_name == splits[0])

        # Episode counts per config — use GoalPredictor (same records for both models)
        gp_cfg = split_config_metrics[split_name].get("GoalPredictor", {})
        config_counts = {cfg: int(m["n"]) for cfg, m in gp_cfg.items()}

        _draw_bar_panel(
            ax,
            config_metrics = split_config_metrics[split_name],
            panel_title    = _SPLIT_TITLE.get(split_name, split_name),
            show_legend    = is_first,
            show_xlabel    = is_last,
            paper          = paper,
            config_counts  = config_counts,
        )

    dpi = 300 if paper else 150
    fig.tight_layout(pad=2.0, rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def print_table(all_metrics: dict[str, dict[str, dict]]) -> None:
    """Print a compact results table to stdout."""
    header = f"  {'Model':<18}  {'Split':<20}  {'N':>5}  {'%Correct':>9}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for model_name, split_metrics in all_metrics.items():
        for split_name, m in split_metrics.items():
            print(
                f"  {model_name:<18}  {split_name:<20}  {m['n']:>5}  "
                f"{m['pct_correct']:>8.1f}%"
            )
    print("─" * len(header))


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate models on distractor episodes."
    )
    p.add_argument("--goal-ckpt",   default="outputs/goal_predictor/best.pt",  metavar="PT")
    p.add_argument("--ctx-ckpt",    default="outputs/context_goal/best.pt",   metavar="PT",
                   help="ContextGoalPredictor checkpoint")
    p.add_argument("--bnk-ckpt",    default="outputs/bdi_bottleneck/best.pt",  metavar="PT",
                   help="BDIBottleneck checkpoint")
    p.add_argument("--splits-dir",  default="data/splits",                     metavar="DIR")
    p.add_argument("--splits",      nargs="+",
                   default=["test", "test_new_agent"])
    p.add_argument("--K",           type=int, default=10, metavar="K")
    p.add_argument("--K-recent",    type=int, default=5,  metavar="K", dest="K_recent")
    p.add_argument("--batch",       type=int, default=64, metavar="N")
    p.add_argument("--no-cuda",     action="store_true")
    p.add_argument("--paper",       action="store_true",
                   help="Paper-ready output: inset labels, no titles, 300 DPI")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    splits_dir = ROOT / args.splits_dir

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
    global poi_to_idx
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    # ── Graph ─────────────────────────────────────────────────────────
    import osmnx as ox
    graph_path = ROOT / "data" / "ucsd_walk.graphml"
    print(f"Loading graph …")
    G           = ox.load_graphml(graph_path)
    node_coords = node_coords_from_graph(G)
    print(f"  {len(node_coords):,} nodes\n")

    # ── Load checkpoints ──────────────────────────────────────────────
    goal_ckpt_path = ROOT / args.goal_ckpt

    goal_ckpt = torch.load(goal_ckpt_path, map_location=device)

    # Infer hyperparameters from checkpoint state dicts
    goal_state = goal_ckpt["model_state"]
    d_model_gp = goal_state["coord_proj.weight"].shape[0]
    num_pois   = goal_state["head.weight"].shape[0]

    goal_model = GoalPredictor(num_pois=num_pois, d_model=d_model_gp).to(device)
    goal_model.load_state_dict(goal_state)
    goal_model.eval()
    print(f"GoalPredictor  loaded from {goal_ckpt_path}")

    ctx_ckpt_path = ROOT / args.ctx_ckpt
    ctx_ckpt      = torch.load(ctx_ckpt_path, map_location=device)
    ctx_state     = ctx_ckpt["model_state"]
    d_model_ctx   = ctx_state["coord_proj.weight"].shape[0]
    ctx_model     = ContextGoalPredictor(num_pois=num_pois, d_model=d_model_ctx).to(device)
    ctx_model.load_state_dict(ctx_state)
    ctx_model.eval()
    print(f"ContextGoalPredictor loaded from {ctx_ckpt_path}")

    bnk_path   = ROOT / args.bnk_ckpt
    bnk_ckpt   = torch.load(bnk_path, map_location=device)
    bnk_state  = bnk_ckpt["model_state"]
    d_model_bn = bnk_state["coord_proj.weight"].shape[0]
    bnk_model  = BDIBottleneckPredictor(num_pois=num_pois, d_model=d_model_bn).to(device)
    bnk_model.load_state_dict(bnk_state)
    bnk_model.eval()
    print(f"BDIBottleneck  loaded from {bnk_path}")
    print()

    # ── Evaluate on each split ─────────────────────────────────────────
    all_metrics: dict[str, dict[str, dict]] = {
        "GoalPredictor":        {},
        "ContextGoalPredictor": {},
        "BDIBottleneck":        {},
    }
    all_preds: dict[str, dict[str, list]] = defaultdict(dict)

    # Per-split, per-config accumulators
    # split_config_preds[split][model][config] = list of pred dicts
    split_config_preds: dict[str, dict[str, dict[str, list]]] = {}

    for split_name in args.splits:
        jsonl_path = splits_dir / f"distractor_{split_name}.jsonl"
        meta_path  = splits_dir / f"distractor_{split_name}_meta.json"

        if not jsonl_path.exists():
            print(f"  WARNING: {jsonl_path} not found — skipping. "
                  f"Run extract_distractor_episodes.py first.")
            continue
        if not meta_path.exists():
            print(f"  WARNING: {meta_path} not found — skipping.")
            continue

        records: list[EpisodeRecord] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EpisodeRecord.from_json(line))

        with open(meta_path) as f:
            meta: list[dict] = json.load(f)

        assert len(records) == len(meta), \
            f"Record/meta length mismatch for {split_name}: {len(records)} vs {len(meta)}"

        print(f"  {split_name}  ({len(records)} distractor episodes)")

        # GoalPredictor
        gp_preds = run_goal_predictor(
            goal_model, records, meta, node_coords, device, args.batch
        )
        all_metrics["GoalPredictor"][split_name] = compute_metrics(gp_preds)
        all_preds["GoalPredictor"][split_name]   = gp_preds

        # ContextGoalPredictor
        ctx_preds = run_ctx_predictor(
            ctx_model, records, meta, node_coords, poi_to_idx, device,
            K=args.K, K_recent=args.K_recent, batch_size=args.batch,
        )
        all_metrics["ContextGoalPredictor"][split_name] = compute_metrics(ctx_preds)
        all_preds["ContextGoalPredictor"][split_name]   = ctx_preds

        # BDIBottleneck
        bn_preds = run_bdi_predictor(
            bnk_model, records, meta, node_coords, poi_to_idx, device,
            K=args.K, K_recent=args.K_recent, batch_size=args.batch,
        )
        all_metrics["BDIBottleneck"][split_name] = compute_metrics(bn_preds)
        all_preds["BDIBottleneck"][split_name]   = bn_preds

        # Accumulate per-config predictions for this split
        cfg_gp:  dict[str, list] = defaultdict(list)
        cfg_ctx: dict[str, list] = defaultdict(list)
        cfg_bn:  dict[str, list] = defaultdict(list)
        for i, (r, gp) in enumerate(zip(records, gp_preds)):
            cfg_gp[r.belief_config].append(gp)
            cfg_ctx[r.belief_config].append(ctx_preds[i])
            cfg_bn[r.belief_config].append(bn_preds[i])
        split_config_preds[split_name] = {
            "GoalPredictor":        cfg_gp,
            "ContextGoalPredictor": cfg_ctx,
            "BDIBottleneck":        cfg_bn,
        }

    # Build per-split, per-config metrics (+ Overall)
    split_config_metrics: dict[str, dict[str, dict[str, dict]]] = {}
    for split_name, model_cfg in split_config_preds.items():
        model_names = list(model_cfg.keys())
        cm: dict[str, dict[str, dict]] = {m: {} for m in model_names}
        for model_name in model_names:
            all_for_split: list[dict] = []
            for cfg in sorted(model_cfg[model_name].keys()):
                preds = model_cfg[model_name][cfg]
                cm[model_name][cfg] = compute_metrics(preds)
                all_for_split.extend(preds)
            cm[model_name]["Overall"] = compute_metrics(all_for_split)
        split_config_metrics[split_name] = cm
        for cfg in sorted(model_cfg["GoalPredictor"].keys()):
            n = len(model_cfg["GoalPredictor"][cfg])
            print(f"    {split_name}  {cfg}: {n} episodes")

    print_table(all_metrics)

    # ── Save results + plots ───────────────────────────────────────────
    out_dir = ROOT / "results" / "distractor_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all metrics as JSON
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "split_metrics":        all_metrics,
            "split_config_metrics": split_config_metrics,
        }, f, indent=2)
    print(f"\n  Results → {out_dir / 'results.json'}")

    # Single stacked figure (Test on top, New Agent on bottom)
    if split_config_metrics:
        fname = "comparison_paper.png" if args.paper else "comparison.png"
        plot_comparison(split_config_metrics, out_dir / fname, paper=args.paper)

    print(f"\n  Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
