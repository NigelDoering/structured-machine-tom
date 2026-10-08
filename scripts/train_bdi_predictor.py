"""
scripts/train_bdi_predictor.py

Train BDIPredictor: a three-stream transformer that jointly attributes
beliefs, desires, and end goal from agent trajectories.

Architecture
------------
  Desire Net  : K random past trajectories (same agent)                → p̂_i ∈ Δ^{M-1}
  Recent Net  : K_recent most-recent past trajectories (same agent)    → z_recent
  Belief head : cat[z_d, z_recent]                                     → b̂_i ∈ [0,1]^M
  Goal Net    : current partial trajectory + z_d + z_recent            → ĝ_i ∈ {1,...,M}

Usage
-----
  .venv/bin/python scripts/train_bdi_predictor.py

Key flags
---------
  --epochs   30         training epochs
  --batch    32         batch size (smaller than baseline due to context overhead)
  --K        10         context set size for both nets
  --lr       3e-4       learning rate
  --lam-g    1.0        goal loss weight
  --lam-d    1.0        desire KL weight
  --lam-b    1.0        belief BCE weight
  --d-model  64         hidden dimension
  --layers   3          encoder depth
  --out      outputs/bdi_predictor

Output
------
  <out>/best.pt
  <out>/goal_accuracy_vs_path_fraction.png
  <out>/training_curves.png
  <out>/accuracy_results.json
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
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model.bdi_predictor import BDIPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.meta_dataset import MetaEpisodeDataset, build_meta_dataloaders, _norm
from simulation.episode import EpisodeRecord


# ── Path-fraction evaluation points ───────────────────────────────────
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BELIEF_LOSS_CHOICES = ("weighted_bce", "balanced_bce", "balanced_focal")


# ──────────────────────────────────────────────────────────────────────
# Random prefix augmentation (applied to current trajectory only)
# ──────────────────────────────────────────────────────────────────────

def random_prefix(
    coords: torch.Tensor,
    mask:   torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate each sample's current trajectory to a random prefix length."""
    B = coords.size(0)
    new_coords = torch.zeros_like(coords)
    new_mask   = torch.zeros_like(mask)
    for i in range(B):
        real_len = int(mask[i].sum().item())
        k = torch.randint(1, real_len + 1, ()).item()
        new_coords[i, :k] = coords[i, :k]
        new_mask[i,   :k] = 1
    return new_coords, new_mask


def belief_loss_with_logits(
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    *,
    mode: str = "weighted_bce",
    belief_weight: float = 5.0,
    focal_gamma: float = 2.0,
    balanced_max_weight: float = 12.0,
) -> torch.Tensor:
    """Compute the belief-head loss with optional class-imbalance variants."""
    if mode not in BELIEF_LOSS_CHOICES:
        raise ValueError(f"Unknown belief loss mode: {mode}")

    weights = torch.ones_like(belief_targets)
    if mode == "weighted_bce":
        weights[belief_targets == 0] = belief_weight
    else:
        closed = (belief_targets == 0).sum().float()
        open_ = (belief_targets == 1).sum().float()
        closed_weight = torch.tensor(1.0, device=belief_targets.device)
        if float(closed.item()) > 0:
            closed_weight = torch.clamp(open_ / closed, min=1.0, max=balanced_max_weight)
        weights[belief_targets == 0] = closed_weight

    bce = F.binary_cross_entropy_with_logits(
        belief_logits, belief_targets, reduction="none"
    )
    if mode == "balanced_focal":
        probs = belief_logits.sigmoid()
        p_t = probs * belief_targets + (1.0 - probs) * (1.0 - belief_targets)
        bce = ((1.0 - p_t).clamp(min=0.0, max=1.0) ** focal_gamma) * bce
    return (weights * bce).mean()


# ──────────────────────────────────────────────────────────────────────
# Training / evaluation
# ──────────────────────────────────────────────────────────────────────

def train_epoch(
    model, loader, optimizer,
    lam_g: float, lam_d: float, lam_b: float,
    belief_weight: float,
    device,
    belief_loss: str = "weighted_bce",
    focal_gamma: float = 2.0,
    balanced_max_weight: float = 12.0,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        dc = batch["desire_coords"].to(device)
        dm = batch["desire_mask"].to(device)
        rc = batch["recent_coords"].to(device)
        rm = batch["recent_mask"].to(device)
        cc, cm = random_prefix(
            batch["current_coords"].to(device),
            batch["current_mask"].to(device),
        )
        goal_labels    = batch["goal_poi_idx"].to(device)
        desire_targets = batch["desire_vector"].to(device)
        belief_targets = batch["belief_vector"].to(device)

        goal_logits, desire_logits, belief_logits = model(dc, dm, rc, rm, cc, cm)

        loss_g = F.cross_entropy(goal_logits, goal_labels)
        loss_d = F.kl_div(
            F.log_softmax(desire_logits, dim=-1), desire_targets, reduction="batchmean"
        )
        loss_b = belief_loss_with_logits(
            belief_logits,
            belief_targets,
            mode=belief_loss,
            belief_weight=belief_weight,
            focal_gamma=focal_gamma,
            balanced_max_weight=balanced_max_weight,
        )

        loss = lam_g * loss_g + lam_d * loss_d + lam_b * loss_b

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * goal_labels.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device) -> dict[str, float]:
    """
    Evaluate on a DataLoader using full current trajectories (no truncation).

    Returns
    -------
    goal_top1   : top-1 accuracy over goal POIs
    belief_acc  : mean per-entry binary accuracy across all M belief entries
                  (reported for reference; inflated by the ~20 always-open entries)
    belief_f1   : F1 score on the false-belief (closed) class — harmonic mean of
                  precision and recall on target=0 entries.  Penalises both missing
                  closed POIs (low recall) and over-predicting closed (low precision).
    desire_kl   : mean KL divergence between predicted and ground-truth desire
    """
    model.eval()
    n_samples         = 0
    goal_correct      = 0
    belief_correct    = 0.0
    fb_tp             = 0   # predicted closed AND truly closed
    fb_fp             = 0   # predicted closed BUT truly open
    fb_fn             = 0   # predicted open  BUT truly closed
    desire_kl_sum     = 0.0

    with torch.no_grad():
        for batch in loader:
            dc = batch["desire_coords"].to(device)
            dm = batch["desire_mask"].to(device)
            rc = batch["recent_coords"].to(device)
            rm = batch["recent_mask"].to(device)
            cc = batch["current_coords"].to(device)
            cm = batch["current_mask"].to(device)

            goal_labels    = batch["goal_poi_idx"].to(device)
            desire_targets = batch["desire_vector"].to(device)
            belief_targets = batch["belief_vector"].to(device)

            goal_logits, desire_logits, belief_logits = model(dc, dm, rc, rm, cc, cm)

            B = goal_labels.size(0)
            n_samples += B

            # Goal top-1
            goal_correct += (goal_logits.argmax(1) == goal_labels).sum().item()

            # Belief overall accuracy (per-entry, averaged over M then over B)
            belief_pred = belief_logits.sigmoid() > 0.5           # (B, M) bool
            belief_correct += (
                (belief_pred == belief_targets.bool()).float().mean(dim=1).sum().item()
            )

            # F1 on false-belief (closed) entries
            true_closed = belief_targets == 0                     # (B, M) bool
            pred_closed = ~belief_pred                            # predicted as closed
            fb_tp += (pred_closed &  true_closed).sum().item()
            fb_fp += (pred_closed & ~true_closed).sum().item()
            fb_fn += (~pred_closed & true_closed).sum().item()

            # Desire KL (summed over samples, divided later)
            desire_kl_sum += F.kl_div(
                F.log_softmax(desire_logits, dim=-1),
                desire_targets,
                reduction="sum",
            ).item()

    precision = fb_tp / (fb_tp + fb_fp) if (fb_tp + fb_fp) > 0 else 0.0
    recall    = fb_tp / (fb_tp + fb_fn) if (fb_tp + fb_fn) > 0 else 0.0
    belief_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "goal_top1":  goal_correct   / n_samples,
        "belief_acc": belief_correct / n_samples,
        "belief_f1":  belief_f1,
        "desire_kl":  desire_kl_sum  / n_samples,
    }


# ──────────────────────────────────────────────────────────────────────
# Context construction helpers for fraction evaluation
# ──────────────────────────────────────────────────────────────────────

def _record_to_coords(r: EpisodeRecord, node_coords: dict) -> torch.Tensor:
    pts = [
        _norm(*node_coords.get(nid, (0.0, 0.0)))
        for nid in r.path
    ]
    return torch.tensor(pts, dtype=torch.float32)


def _build_context_tensor(
    records_ctx: list[EpisodeRecord],
    node_coords: dict,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (K, T_max, 2) and (K, T_max) from a list of context records."""
    tensors = [_record_to_coords(r, node_coords) for r in records_ctx]
    # Pad or repeat to exactly K
    while len(tensors) < K:
        tensors.append(tensors[-1].clone() if tensors else torch.zeros(1, 2))
    tensors = tensors[:K]
    T_max = max(t.size(0) for t in tensors)
    coords = torch.zeros(K, T_max, 2)
    mask   = torch.zeros(K, T_max, dtype=torch.long)
    for i, t in enumerate(tensors):
        T = t.size(0)
        coords[i, :T] = t
        mask[i,   :T] = 1
    return coords, mask


def _build_agent_index(
    records: list[EpisodeRecord],
) -> dict[str, list[int]]:
    """Map agent_id → ordered list of flat record indices (simulation order)."""
    idx_map: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        idx_map[r.agent_id].append(i)
    return idx_map


# ──────────────────────────────────────────────────────────────────────
# Fraction evaluation
# ──────────────────────────────────────────────────────────────────────

def eval_at_fractions(
    model,
    records:     list[EpisodeRecord],
    node_coords: dict,
    poi_to_idx:  dict[str, int],
    device,
    K:           int = 10,
    K_recent:    int = 5,
    batch_size:  int = 64,
    rng_seed:    int = 42,
) -> dict[float, dict[str, float]]:
    """
    Evaluate top-1 and top-5 goal accuracy at each path fraction in FRACTIONS.
    Context trajectories are constructed deterministically (fixed seed).
    """
    model.eval()
    rng       = __import__("numpy").random.default_rng(rng_seed)
    agent_idx = _build_agent_index(records)
    results: dict[float, dict[str, float]] = {}

    for frac in tqdm(FRACTIONS, desc="  fractions", leave=False, unit="frac"):
        correct1 = 0
        correct5 = 0

        for start in range(0, len(records), batch_size):
            batch_recs = records[start : start + batch_size]
            B = len(batch_recs)

            desire_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
            recent_ctx_list: list[tuple[torch.Tensor, torch.Tensor]] = []
            curr_list:       list[torch.Tensor]                      = []
            labels:          list[int]                               = []

            for r in batch_recs:
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

                # Recent context: K_recent most-recent preceding trajectories
                preceding = agent_idxs[:pos]
                if len(preceding) == 0:
                    fallback = agent_idxs[pos + 1 :] or [flat_idx]
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

                # Current trajectory truncated to fraction
                T = len(r.path)
                k = max(1, round(frac * T))
                pts = [
                    _norm(*node_coords.get(nid, (0.0, 0.0)))
                    for nid in r.path[:k]
                ]
                curr_list.append(torch.tensor(pts, dtype=torch.float32))
                labels.append(poi_to_idx[r.goal_poi])

            # ── Pad and batch ─────────────────────────────────────
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

            dc, dm = stack_context(desire_ctx_list)
            rc, rm = stack_context(recent_ctx_list)

            T_c = max(c.size(0) for c in curr_list)
            cc  = torch.zeros(B, T_c, 2)
            cm  = torch.zeros(B, T_c, dtype=torch.long)
            for i, c in enumerate(curr_list):
                cc[i, :c.size(0)] = c
                cm[i, :c.size(0)] = 1

            labels_t = torch.tensor(labels)

            with torch.no_grad():
                goal_logits, _, _ = model(
                    dc.to(device), dm.to(device),
                    rc.to(device), rm.to(device),
                    cc.to(device), cm.to(device),
                )

            top5 = goal_logits.topk(5, dim=1).indices.cpu()
            correct1 += (top5[:, 0] == labels_t).sum().item()
            correct5 += (top5 == labels_t.unsqueeze(1)).any(1).sum().item()

        n = len(records)
        results[frac] = {"top1": correct1 / n, "top5": correct5 / n}

    return results


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

_SPLIT_STYLE = {
    "val":            {"color": "#2196F3", "label": "Val"},
    "test":           {"color": "#FF9800", "label": "Test"},
    "test_new_agent": {"color": "#9C27B0", "label": "Test (new agent)"},
}


def plot_goal_accuracy(
    split_results: dict[str, dict[float, dict[str, float]]],
    out_dir: Path,
) -> None:
    """Top-1 and top-5 goal accuracy vs path fraction — one line per split."""
    pct = [int(f * 100) for f in FRACTIONS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(
        axes, ["top1", "top5"], ["Top-1 Goal Accuracy", "Top-5 Goal Accuracy"]
    ):
        for split_name, res in split_results.items():
            style = _SPLIT_STYLE.get(split_name, {"color": "gray", "label": split_name})
            ys = [res[f][metric] * 100 for f in FRACTIONS]
            ax.plot(pct, ys, marker="o", linewidth=2, markersize=5,
                    color=style["color"], label=style["label"])
        ax.set_xlabel("Path revealed (%)", fontsize=12)
        ax.set_ylabel("Accuracy (%)",      fontsize=12)
        ax.set_title(title,                fontsize=13, fontweight="bold")
        ax.set_xlim(5, 105)
        ax.set_ylim(0, 105)
        ax.set_xticks(pct)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    path = out_dir / "goal_accuracy_vs_path_fraction.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_training_curves(
    history: dict[str, list[float]],
    out_dir: Path,
) -> None:
    """
    Three-panel figure: belief overall accuracy, false-belief F1, desire KL.

    belief_acc is shown for reference but is inflated by the majority open class.
    belief_f1 is the headline metric: harmonic mean of precision and recall on
    the rare closed entries, penalising both missed false beliefs and over-prediction.

    history keys: "belief_acc", "belief_f1", "desire_kl"
    """
    epochs = list(range(1, len(history["belief_acc"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Belief overall accuracy
    ax = axes[0]
    ax.plot(epochs, [v * 100 for v in history["belief_acc"]],
            color="#E91E63", linewidth=2)
    ax.set_xlabel("Epoch",                    fontsize=12)
    ax.set_ylabel("Accuracy (%)",             fontsize=12)
    ax.set_title("Val Belief Accuracy\n(all entries)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: False-belief F1
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in history["belief_f1"]],
            color="#FF5722", linewidth=2)
    ax.set_xlabel("Epoch",                     fontsize=12)
    ax.set_ylabel("F1 (%)",                    fontsize=12)
    ax.set_title("Val False-Belief F1\n(closed entries only)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: Desire KL divergence
    ax = axes[2]
    ax.plot(epochs, history["desire_kl"],
            color="#4CAF50", linewidth=2)
    ax.set_xlabel("Epoch",                   fontsize=12)
    ax.set_ylabel("KL Divergence",           fontsize=12)
    ax.set_title("Val Desire KL Divergence", fontsize=13, fontweight="bold")
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────
# CLI + main
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BDIPredictor on BDI ToM meta-episodes."
    )
    p.add_argument("--epochs",  type=int,   default=30,   metavar="N")
    p.add_argument("--batch",   type=int,   default=32,   metavar="N")
    p.add_argument("--K",        type=int,   default=10,   metavar="K",
                   help="Context set size for desire net")
    p.add_argument("--K-recent", type=int,   default=5,    metavar="K",
                   dest="K_recent",
                   help="Context set size for recent net (belief comparative signal)")
    p.add_argument("--lr",      type=float, default=3e-4, metavar="LR")
    p.add_argument("--wd",      type=float, default=1e-4, metavar="WD")
    p.add_argument("--lam-g",   type=float, default=1.0,  metavar="W",
                   dest="lam_g", help="Goal loss weight")
    p.add_argument("--lam-d",   type=float, default=1.0,  metavar="W",
                   dest="lam_d", help="Desire KL weight")
    p.add_argument("--lam-b",   type=float, default=5.0,  metavar="W",
                   dest="lam_b", help="Belief BCE weight")
    p.add_argument("--belief-weight", type=float, default=5.0, metavar="W",
                   dest="belief_weight",
                   help="BCE loss multiplier on false-belief entries (target=0); "
                        "corrects class imbalance between open (~20) and closed (~4) entries")
    p.add_argument("--belief-loss", choices=BELIEF_LOSS_CHOICES, default="weighted_bce",
                   help="Belief loss mode for false-belief imbalance diagnostics")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Gamma for --belief-loss balanced_focal")
    p.add_argument("--balanced-max-weight", type=float, default=12.0,
                   help="Max closed-entry class weight for balanced belief losses")
    p.add_argument("--d-model", type=int,   default=64,   metavar="D",
                   dest="d_model")
    p.add_argument("--layers",  type=int,   default=3,    metavar="N")
    p.add_argument("--heads",   type=int,   default=4,    metavar="N")
    p.add_argument("--dim-ff",  type=int,   default=256,  metavar="D",
                   dest="dim_ff")
    p.add_argument("--warmup",  type=int,   default=0,    metavar="N",
                   help="Linear LR warmup epochs before cosine decay (0 = disabled)")
    p.add_argument("--dropout", type=float, default=0.1,  metavar="P")
    p.add_argument("--splits",  type=str,   default="data/splits",  metavar="DIR")
    p.add_argument("--out",     type=str,   default="outputs/bdi_predictor", metavar="DIR")
    p.add_argument("--no-cuda", action="store_true")
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
    out_dir    = ROOT / args.out
    splits_dir = ROOT / args.splits
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────
    if args.no_cuda:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    # ── Configs ──────────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)

    poi_order = [
        poi["id"]
        for cat_data in poi_cfg["categories"].values()
        for poi in cat_data["pois"]
    ]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    # ── Graph ─────────────────────────────────────────────────────────
    graph_path = ROOT / "data" / "ucsd_walk.graphml"
    try:
        import osmnx as ox
        print(f"Loading graph from {graph_path}")
        G = ox.load_graphml(graph_path)
        node_coords = node_coords_from_graph(G)
        print(f"Graph loaded  ({len(node_coords):,} nodes)")
    except Exception as exc:
        print(f"ERROR loading graph: {exc}")
        sys.exit(1)

    # ── DataLoaders ───────────────────────────────────────────────────
    print("\nBuilding meta-episode DataLoaders...")
    loaders = build_meta_dataloaders(
        splits_dir  = splits_dir,
        node_coords = node_coords,
        poi_to_idx  = poi_to_idx,
        K           = args.K,
        K_recent    = args.K_recent,
        batch_size  = args.batch,
        splits      = ["train", "val"],
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ── Raw records for fraction evaluation ───────────────────────────
    eval_splits: dict[str, list[EpisodeRecord]] = {}
    for name in ("val", "test", "test_new_agent"):
        p = splits_dir / f"{name}.jsonl"
        if p.exists():
            eval_splits[name] = _load_records(p)
            print(f"  Loaded {len(eval_splits[name]):>5} records  ← {name}")

    # ── Model ─────────────────────────────────────────────────────────
    model = BDIPredictor(
        num_pois   = 24,
        d_model    = args.d_model,
        nhead      = args.heads,
        num_layers = args.layers,
        dim_ff     = args.dim_ff,
        dropout    = args.dropout,
    ).to(device)


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBDIPredictor  —  {n_params:,} trainable parameters")
    print(f"  d_model={args.d_model}, heads={args.heads}, layers={args.layers}, "
          f"dim_ff={args.dim_ff}, K={args.K}, K_recent={args.K_recent}")
    print(f"  λ_goal={args.lam_g}, λ_desire={args.lam_d}, λ_belief={args.lam_b}, "
          f"belief_loss={args.belief_loss}, belief_weight={args.belief_weight}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.warmup > 0:
        warmup_sched = LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup
        )
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.warmup), eta_min=args.lr / 10
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    # PyTorch SequentialLR internally passes an epoch argument to sub-schedulers,
    # triggering a deprecation warning that is harmless and not actionable from user code.
    warnings.filterwarnings(
        "ignore",
        message="The epoch parameter in `scheduler.step\\(\\)`",
        category=UserWarning,
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_goal = -1.0
    best_ckpt     = out_dir / "best.pt"

    history = {"belief_acc": [], "belief_f1": [], "desire_kl": [], "goal_top1": []}

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss = train_epoch(
            model, train_loader, optimizer,
            args.lam_g, args.lam_d, args.lam_b, args.belief_weight, device,
            args.belief_loss, args.focal_gamma, args.balanced_max_weight,
        )
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        history["goal_top1"].append(val_metrics["goal_top1"])
        history["belief_acc"].append(val_metrics["belief_acc"])
        history["belief_f1"].append(val_metrics["belief_f1"])
        history["desire_kl"].append(val_metrics["desire_kl"])

        is_best = val_metrics["goal_top1"] > best_val_goal
        if is_best:
            best_val_goal = val_metrics["goal_top1"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_metrics": val_metrics},
                best_ckpt,
            )

        epoch_bar.set_postfix(
            tr_loss    = f"{tr_loss:.4f}",
            goal_top1  = f"{val_metrics['goal_top1'] * 100:.1f}%",
            belief_acc = f"{val_metrics['belief_acc'] * 100:.1f}%",
            belief_f1  = f"{val_metrics['belief_f1'] * 100:.1f}%",
            desire_kl  = f"{val_metrics['desire_kl']:.4f}",
            best       = f"{best_val_goal * 100:.1f}%" + (" ✓" if is_best else ""),
        )

    print(f"\nBest val goal top-1: {best_val_goal * 100:.1f}%")
    print(f"Checkpoint → {best_ckpt}")

    # ── Load best checkpoint ──────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ── Goal accuracy at 10% path intervals ───────────────────────────
    print("\n" + "=" * 57)
    print("  Evaluating goal accuracy at 10% path intervals")
    print("=" * 57)

    split_results: dict[str, dict[float, dict[str, float]]] = {}
    for name, records in tqdm(
        eval_splits.items(), desc="Evaluating splits", unit="split"
    ):
        split_results[name] = eval_at_fractions(
            model, records, node_coords, poi_to_idx, device,
            K=args.K, K_recent=args.K_recent,
        )
        tqdm.write(f"\n  {name}  ({len(records)} episodes)")
        tqdm.write(f"  {'Frac':>6}  {'Top-1':>7}  {'Top-5':>7}")
        for frac, vals in split_results[name].items():
            tqdm.write(
                f"  {frac * 100:>5.0f}%  {vals['top1'] * 100:>6.1f}%"
                f"  {vals['top5'] * 100:>6.1f}%"
            )

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nSaving plots...")
    plot_goal_accuracy(split_results, out_dir)
    plot_training_curves(history, out_dir)

    # Save all results
    results_path = out_dir / "accuracy_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "goal_fractions": {
                    s: {str(k): v for k, v in r.items()}
                    for s, r in split_results.items()
                },
                "training_history": history,
            },
            f,
            indent=2,
        )
    print(f"  Results → {results_path}")


if __name__ == "__main__":
    main()
