"""
scripts/train_bdi_bottleneck_ablation.py

Train ablation variants of BDIBottleneckPredictor that remove one mental-state
channel to isolate each component's contribution.

Ablations
---------
  no_desire : removes desire encoder + desire head
              trains with λ_g * L_goal + λ_b * L_belief
              model forward: (recent_coords, recent_mask, current_coords, current_mask)
              model returns: (goal_logits, belief_logits)

  no_belief : removes recent encoder + belief head
              trains with λ_g * L_goal + λ_d * L_desire
              model forward: (desire_coords, desire_mask, current_coords, current_mask)
              model returns: (goal_logits, desire_logits)

Usage
-----
  .venv/bin/python scripts/train_bdi_bottleneck_ablation.py --ablation no_desire
  .venv/bin/python scripts/train_bdi_bottleneck_ablation.py --ablation no_belief

Key flags
---------
  --ablation  {no_desire,no_belief}   required
  --epochs    30
  --batch     32
  --K         10    context size for desire (no_belief) or recent (no_desire)
  --K-recent   5    recent context size (only used by no_desire)
  --lr        3e-4
  --lam-g     1.0
  --lam-d     1.0   (no_belief only)
  --lam-b     5.0   (no_desire only)
  --seed      None  random seed; outputs go to <out>/seed_<N>/
  --out       outputs/ablation_{ablation}

Output
------
  <out>/best.pt
  <out>/training_curves.png
  <out>/accuracy_results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model.bdi_bottleneck_ablations import BDIBottleneckNoDesire, BDIBottleneckNoBelief
from model.dataset import build_poi_index, node_coords_from_graph
from model.meta_dataset import build_meta_dataloaders, _norm
from simulation.episode import EpisodeRecord
from scripts.train_bdi_predictor import (
    FRACTIONS,
    random_prefix,
    _load_records,
    _record_to_coords,
    _build_context_tensor,
    _build_agent_index,
)
from scripts.diag_utils import adamw_kwargs, maybe_force_math_sdp


# ── Training ───────────────────────────────────────────────────────────────


def train_epoch_ablation(
    model,
    loader,
    optimizer,
    ablation:      str,
    lam_g:         float,
    lam_d:         float,
    lam_b:         float,
    belief_weight: float,
    device,
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

        if ablation == "no_desire":
            goal_logits, belief_logits = model(rc, rm, cc, cm)
            loss_g = F.cross_entropy(goal_logits, goal_labels)
            bce_w  = torch.ones_like(belief_targets)
            bce_w[belief_targets == 0] = belief_weight
            loss_b = F.binary_cross_entropy_with_logits(
                belief_logits, belief_targets, weight=bce_w
            )
            loss = lam_g * loss_g + lam_b * loss_b

        else:  # no_belief
            goal_logits, desire_logits = model(dc, dm, cc, cm)
            loss_g = F.cross_entropy(goal_logits, goal_labels)
            loss_d = F.kl_div(
                F.log_softmax(desire_logits, dim=-1),
                desire_targets,
                reduction="batchmean",
            )
            loss = lam_g * loss_g + lam_d * loss_d

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * goal_labels.size(0)

    return total_loss / len(loader.dataset)


# ── Epoch evaluation ───────────────────────────────────────────────────────


def eval_epoch_ablation(model, loader, ablation: str, device) -> dict[str, float]:
    model.eval()
    n_samples     = 0
    goal_correct  = 0
    # belief metrics (no_desire only)
    belief_correct = 0.0
    fb_tp = fb_fp = fb_fn = 0
    # desire metric (no_belief only)
    desire_kl_sum = 0.0

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

            if ablation == "no_desire":
                goal_logits, belief_logits = model(rc, rm, cc, cm)
                B = goal_labels.size(0)
                n_samples += B
                goal_correct += (goal_logits.argmax(1) == goal_labels).sum().item()

                belief_pred  = belief_logits.sigmoid() > 0.5
                belief_correct += (
                    (belief_pred == belief_targets.bool()).float().mean(dim=1).sum().item()
                )
                true_closed = belief_targets == 0
                pred_closed = ~belief_pred
                fb_tp += (pred_closed &  true_closed).sum().item()
                fb_fp += (pred_closed & ~true_closed).sum().item()
                fb_fn += (~pred_closed & true_closed).sum().item()

            else:  # no_belief
                goal_logits, desire_logits = model(dc, dm, cc, cm)
                B = goal_labels.size(0)
                n_samples += B
                goal_correct += (goal_logits.argmax(1) == goal_labels).sum().item()

                desire_kl_sum += F.kl_div(
                    F.log_softmax(desire_logits, dim=-1),
                    desire_targets,
                    reduction="sum",
                ).item()

    metrics: dict[str, float] = {"goal_top1": goal_correct / n_samples}

    if ablation == "no_desire":
        precision = fb_tp / (fb_tp + fb_fp) if (fb_tp + fb_fp) > 0 else 0.0
        recall    = fb_tp / (fb_tp + fb_fn) if (fb_tp + fb_fn) > 0 else 0.0
        belief_f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        metrics["belief_acc"] = belief_correct / n_samples
        metrics["belief_f1"]  = belief_f1
    else:
        metrics["desire_kl"] = desire_kl_sum / n_samples

    return metrics


# ── Fraction evaluation ────────────────────────────────────────────────────


def eval_at_fractions_ablation(
    model,
    records:     list[EpisodeRecord],
    node_coords: dict,
    poi_to_idx:  dict[str, int],
    ablation:    str,
    device,
    K:           int = 10,
    K_recent:    int = 5,
    batch_size:  int = 64,
    rng_seed:    int = 42,
) -> dict[float, dict[str, float]]:
    """Top-1 and top-5 goal accuracy at each path fraction for an ablation model."""
    model.eval()
    rng       = np.random.default_rng(rng_seed)
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

                # Desire context (used by no_belief)
                if ablation == "no_belief":
                    if len(others) == 0:
                        d_src = [flat_idx] * K
                    elif len(others) < K:
                        d_src = rng.choice(others, size=K, replace=True).tolist()
                    else:
                        d_src = rng.choice(others, size=K, replace=False).tolist()
                    desire_ctx_list.append(
                        _build_context_tensor([records[i] for i in d_src], node_coords, K)
                    )

                # Recent context (used by no_desire)
                if ablation == "no_desire":
                    preceding = agent_idxs[:pos]
                    if len(preceding) == 0:
                        fallback = agent_idxs[pos + 1 :] or [flat_idx]
                        r_src = (fallback * ((K_recent // len(fallback)) + 1))[:K_recent]
                    elif len(preceding) < K_recent:
                        r_src = (preceding * ((K_recent // len(preceding)) + 1))[:K_recent]
                    else:
                        r_src = preceding[-K_recent:]
                    recent_ctx_list.append(
                        _build_context_tensor(
                            [records[i] for i in r_src], node_coords, K_recent
                        )
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

            # ── Pad and batch ─────────────────────────────────────────
            def stack_context(ctx_list, dim2):
                K_  = ctx_list[0][0].size(0)
                T_  = max(c.size(1) for c, _ in ctx_list)
                B_  = len(ctx_list)
                co  = torch.zeros(B_, K_, T_, dim2)
                ma  = torch.zeros(B_, K_, T_, dtype=torch.long)
                for i, (c, m) in enumerate(ctx_list):
                    t = c.size(1)
                    co[i, :, :t, :] = c
                    ma[i, :, :t]    = m
                return co, ma

            T_c = max(c.size(0) for c in curr_list)
            cc  = torch.zeros(B, T_c, 2)
            cm  = torch.zeros(B, T_c, dtype=torch.long)
            for i, c in enumerate(curr_list):
                cc[i, :c.size(0)] = c
                cm[i, :c.size(0)] = 1

            labels_t = torch.tensor(labels)

            with torch.no_grad():
                if ablation == "no_desire":
                    rc, rm = stack_context(recent_ctx_list, 2)
                    goal_logits, _ = model(
                        rc.to(device), rm.to(device),
                        cc.to(device), cm.to(device),
                    )
                else:  # no_belief
                    dc, dm = stack_context(desire_ctx_list, 2)
                    goal_logits, _ = model(
                        dc.to(device), dm.to(device),
                        cc.to(device), cm.to(device),
                    )

            top5 = goal_logits.topk(5, dim=1).indices.cpu()
            correct1 += (top5[:, 0] == labels_t).sum().item()
            correct5 += (top5 == labels_t.unsqueeze(1)).any(1).sum().item()

        n = len(records)
        results[frac] = {"top1": correct1 / n, "top5": correct5 / n}

    return results


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_training_curves_ablation(
    history:  dict[str, list[float]],
    ablation: str,
    out_dir:  Path,
) -> None:
    epochs = list(range(1, len(history["goal_top1"]) + 1))

    if ablation == "no_desire":
        panels = [
            ("goal_top1",  "Goal Top-1 (%)",        lambda v: v * 100),
            ("belief_acc", "Belief Accuracy (%)",    lambda v: v * 100),
            ("belief_f1",  "False-Belief F1 (%)",    lambda v: v * 100),
        ]
    else:
        panels = [
            ("goal_top1", "Goal Top-1 (%)",       lambda v: v * 100),
            ("desire_kl", "Desire KL Divergence", lambda v: v),
        ]

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    colors = ["#2196F3", "#E91E63", "#FF5722", "#4CAF50"]
    for ax, (key, ylabel, transform), color in zip(axes, panels, colors):
        ax.plot(epochs, [transform(v) for v in history[key]],
                color=color, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Val {ylabel}", fontsize=13, fontweight="bold")
        ax.set_xlim(1, max(epochs))
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BDIBottleneck ablation (no_desire or no_belief)."
    )
    p.add_argument(
        "--ablation", required=True, choices=["no_desire", "no_belief"],
        help="Which mental-state channel to remove",
    )
    p.add_argument("--epochs",  type=int,   default=30,   metavar="N")
    p.add_argument("--batch",   type=int,   default=32,   metavar="N")
    p.add_argument("--K",        type=int,   default=10,   metavar="K")
    p.add_argument("--K-recent", type=int,   default=5,    metavar="K", dest="K_recent")
    p.add_argument("--lr",      type=float, default=3e-4, metavar="LR")
    p.add_argument("--wd",      type=float, default=1e-4, metavar="WD")
    p.add_argument("--lam-g",   type=float, default=1.0,  metavar="W", dest="lam_g")
    p.add_argument("--lam-d",   type=float, default=1.0,  metavar="W", dest="lam_d",
                   help="Desire KL weight (no_belief only)")
    p.add_argument("--lam-b",   type=float, default=5.0,  metavar="W", dest="lam_b",
                   help="Belief BCE weight (no_desire only)")
    p.add_argument("--belief-weight", type=float, default=5.0, metavar="W",
                   dest="belief_weight",
                   help="BCE loss multiplier on false-belief entries (no_desire only)")
    p.add_argument("--d-model", type=int,   default=64,   metavar="D", dest="d_model")
    p.add_argument("--layers",  type=int,   default=3,    metavar="N")
    p.add_argument("--heads",   type=int,   default=4,    metavar="N")
    p.add_argument("--dim-ff",  type=int,   default=256,  metavar="D", dest="dim_ff")
    p.add_argument("--warmup",  type=int,   default=0,    metavar="N")
    p.add_argument("--dropout", type=float, default=0.1,  metavar="P")
    p.add_argument("--splits",  type=str,   default="data/splits",  metavar="DIR")
    p.add_argument("--out",     type=str,   default=None, metavar="DIR",
                   help="Output directory (default: outputs/ablation_<ablation>)")
    p.add_argument("--seed",    type=int,   default=None, metavar="N",
                   help="Random seed; outputs written to <out>/seed_<N>/")
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    base_dir = ROOT / (args.out or f"outputs/ablation_{args.ablation}")
    out_dir  = base_dir / f"seed_{args.seed}" if args.seed is not None else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = ROOT / args.splits

    # ── Device ───────────────────────────────────────────────────────
    if args.no_cuda:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device  : {device}")
    if device.type == "cuda" and maybe_force_math_sdp():
        print("BDI_FORCE_MATH_SDP=1: using math SDPA backend")
    print(f"Ablation: {args.ablation}")

    # ── Config & graph ────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order  = [poi["id"] for cat_data in poi_cfg["categories"].values()
                  for poi in cat_data["pois"]]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    import osmnx as ox
    graph_path  = ROOT / "data" / "ucsd_walk.graphml"
    G           = ox.load_graphml(graph_path)
    node_coords = node_coords_from_graph(G)
    print(f"Graph loaded  ({len(node_coords):,} nodes)")

    # ── DataLoaders ───────────────────────────────────────────────────
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
    model_kwargs = dict(
        num_pois   = 24,
        d_model    = args.d_model,
        nhead      = args.heads,
        num_layers = args.layers,
        dim_ff     = args.dim_ff,
        dropout    = args.dropout,
    )
    if args.ablation == "no_desire":
        model = BDIBottleneckNoDesire(**model_kwargs).to(device)
    else:
        model = BDIBottleneckNoBelief(**model_kwargs).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = type(model).__name__
    print(f"\n{model_name}  —  {n_params:,} trainable parameters")

    opt_kwargs = adamw_kwargs()
    if opt_kwargs.get("foreach") is False:
        print("BDI_DISABLE_FOREACH_OPT=1: disabling AdamW foreach kernels")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, **opt_kwargs)
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

    warnings.filterwarnings(
        "ignore",
        message="The epoch parameter in `scheduler.step\\(\\)`",
        category=UserWarning,
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_goal = -1.0
    best_ckpt     = out_dir / "best.pt"

    history_keys = (
        ["goal_top1", "belief_acc", "belief_f1"]
        if args.ablation == "no_desire"
        else ["goal_top1", "desire_kl"]
    )
    history: dict[str, list[float]] = {k: [] for k in history_keys}

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss = train_epoch_ablation(
            model, train_loader, optimizer,
            args.ablation, args.lam_g, args.lam_d, args.lam_b,
            args.belief_weight, device,
        )
        val_metrics = eval_epoch_ablation(model, val_loader, args.ablation, device)
        scheduler.step()

        for k in history_keys:
            history[k].append(val_metrics[k])

        is_best = val_metrics["goal_top1"] > best_val_goal
        if is_best:
            best_val_goal = val_metrics["goal_top1"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_metrics": val_metrics, "ablation": args.ablation,
                 "model_kwargs": model_kwargs},
                best_ckpt,
            )

        postfix = {
            "tr_loss":   f"{tr_loss:.4f}",
            "goal_top1": f"{val_metrics['goal_top1'] * 100:.1f}%",
            "best":      f"{best_val_goal * 100:.1f}%" + (" ✓" if is_best else ""),
        }
        if args.ablation == "no_desire":
            postfix["belief_f1"] = f"{val_metrics['belief_f1'] * 100:.1f}%"
        else:
            postfix["desire_kl"] = f"{val_metrics['desire_kl']:.4f}"
        epoch_bar.set_postfix(**postfix)

    print(f"\nBest val goal top-1: {best_val_goal * 100:.1f}%")
    print(f"Checkpoint → {best_ckpt}")

    # ── Load best checkpoint ──────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ── Goal accuracy at path intervals ───────────────────────────────
    print("\n" + "=" * 57)
    print("  Evaluating goal accuracy at 10% path intervals")
    print("=" * 57)

    split_results: dict[str, dict[float, dict[str, float]]] = {}
    for name, records in tqdm(eval_splits.items(), desc="Evaluating splits", unit="split"):
        split_results[name] = eval_at_fractions_ablation(
            model, records, node_coords, poi_to_idx, args.ablation, device,
            K=args.K, K_recent=args.K_recent,
        )
        tqdm.write(f"\n  {name}  ({len(records)} episodes)")
        tqdm.write(f"  {'Frac':>6}  {'Top-1':>7}  {'Top-5':>7}")
        for frac, vals in split_results[name].items():
            tqdm.write(
                f"  {frac * 100:>5.0f}%  {vals['top1'] * 100:>6.1f}%"
                f"  {vals['top5'] * 100:>6.1f}%"
            )

    # ── Plots & results ───────────────────────────────────────────────
    print("\nSaving plots...")
    plot_training_curves_ablation(history, args.ablation, out_dir)

    with open(out_dir / "accuracy_results.json", "w") as f:
        json.dump(
            {
                "ablation": args.ablation,
                "goal_fractions": {
                    s: {str(k): v for k, v in r.items()}
                    for s, r in split_results.items()
                },
                "training_history": history,
            },
            f, indent=2,
        )
    print(f"  Results → {out_dir / 'accuracy_results.json'}")


if __name__ == "__main__":
    main()
