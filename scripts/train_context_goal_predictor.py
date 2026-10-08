"""
scripts/train_context_goal_predictor.py

Train ContextGoalPredictor: the agent-aware goal prediction baseline.

This model has the same three-stream transformer architecture as
BDIBottleneckPredictor but omits the belief and desire prediction heads.
It is trained with a single cross-entropy loss on the goal POI.

Role in the comparison
----------------------
  GoalPredictor          : no agent context  →  lower bound
  ContextGoalPredictor   : agent context, no belief/desire structure  ← this
  BDIBottleneckPredictor : agent context + structured mental states   ← full model

Any accuracy gap between ContextGoalPredictor and BDIBottleneckPredictor is
attributable specifically to the structured belief/desire bottleneck, since
both models have identical encoder architectures and identical access to
agent-stratified past trajectories.

Usage
-----
  .venv/bin/python scripts/train_context_goal_predictor.py
  .venv/bin/python scripts/train_context_goal_predictor.py --epochs 30 --out outputs/context_goal

Key flags
---------
  --epochs    30
  --batch     32
  --K         10
  --K-recent   5
  --lr        3e-4
  --d-model   64
  --layers     3
  --out       outputs/context_goal

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
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model.context_goal_predictor import ContextGoalPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.meta_dataset import build_meta_dataloaders
from simulation.episode import EpisodeRecord

# ── Re-use context construction and plotting utilities ────────────────
from scripts.train_bdi_predictor import (
    FRACTIONS,
    random_prefix,
    plot_goal_accuracy,
    plot_training_curves,
    _load_records,
    _build_agent_index,
    _build_context_tensor,
    _norm,
)
from scripts.diag_utils import adamw_kwargs, maybe_force_math_sdp


# ── Training ───────────────────────────────────────────────────────────

def train_epoch(
    model:     ContextGoalPredictor,
    loader,
    optimizer,
    device,
) -> float:
    """Single cross-entropy loss on goal POI — no auxiliary heads."""
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
        goal_labels = batch["goal_poi_idx"].to(device)

        goal_logits = model(dc, dm, rc, rm, cc, cm)
        loss        = F.cross_entropy(goal_logits, goal_labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * goal_labels.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(
    model:  ContextGoalPredictor,
    loader,
    device,
) -> dict[str, float]:
    """Evaluate top-1 goal accuracy on a DataLoader (full trajectories)."""
    model.eval()
    n_samples    = 0
    goal_correct = 0

    with torch.no_grad():
        for batch in loader:
            dc = batch["desire_coords"].to(device)
            dm = batch["desire_mask"].to(device)
            rc = batch["recent_coords"].to(device)
            rm = batch["recent_mask"].to(device)
            cc = batch["current_coords"].to(device)
            cm = batch["current_mask"].to(device)

            goal_labels = batch["goal_poi_idx"].to(device)
            goal_logits = model(dc, dm, rc, rm, cc, cm)

            n_samples    += goal_labels.size(0)
            goal_correct += (goal_logits.argmax(1) == goal_labels).sum().item()

    return {"goal_top1": goal_correct / n_samples}


# ── Fraction evaluation ────────────────────────────────────────────────

def eval_at_fractions(
    model:       ContextGoalPredictor,
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
    Constructs agent-stratified context tensors (desire + recent) per episode.
    """
    import numpy as np

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

                T   = len(r.path)
                k   = max(1, round(frac * T))
                pts = [_norm(*node_coords.get(nid, (0.0, 0.0))) for nid in r.path[:k]]
                curr_list.append(torch.tensor(pts, dtype=torch.float32))
                labels.append(poi_to_idx[r.goal_poi])

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
                goal_logits = model(
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


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ContextGoalPredictor (agent-context baseline, no belief/desire heads)."
    )
    p.add_argument("--epochs",   type=int,   default=30,              metavar="N")
    p.add_argument("--batch",    type=int,   default=32,              metavar="N")
    p.add_argument("--K",        type=int,   default=10,              metavar="K")
    p.add_argument("--K-recent", type=int,   default=5,               metavar="K", dest="K_recent")
    p.add_argument("--lr",       type=float, default=3e-4,            metavar="LR")
    p.add_argument("--wd",       type=float, default=1e-4,            metavar="WD")
    p.add_argument("--d-model",  type=int,   default=64,              metavar="D", dest="d_model")
    p.add_argument("--layers",   type=int,   default=3,               metavar="N")
    p.add_argument("--heads",    type=int,   default=4,               metavar="N")
    p.add_argument("--dim-ff",   type=int,   default=256,             metavar="D", dest="dim_ff")
    p.add_argument("--warmup",   type=int,   default=0,               metavar="N")
    p.add_argument("--dropout",  type=float, default=0.1,             metavar="P")
    p.add_argument("--splits",   type=str,   default="data/splits",   metavar="DIR")
    p.add_argument("--out",      type=str,   default="outputs/context_goal", metavar="DIR")
    p.add_argument("--seed",     type=int,   default=None,           metavar="N",
                   help="Random seed for training (if set, writes to out/seed_<N>)")
    p.add_argument("--no-cuda",  action="store_true")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    out_dir    = ROOT / args.out
    if args.seed is not None:
        out_dir = out_dir / f"seed_{args.seed}"
    splits_dir = ROOT / args.splits
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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
    if device.type == "cuda" and maybe_force_math_sdp():
        print("BDI_FORCE_MATH_SDP=1: using math SDPA backend")

    # ── POI config ───────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order  = [poi["id"] for cat_data in poi_cfg["categories"].values()
                  for poi in cat_data["pois"]]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    # ── Graph ────────────────────────────────────────────────────────
    import osmnx as ox
    graph_path = ROOT / "data" / "ucsd_walk.graphml"
    print(f"Loading graph from {graph_path}")
    G           = ox.load_graphml(graph_path)
    node_coords = node_coords_from_graph(G)
    print(f"Graph loaded  ({len(node_coords):,} nodes)")

    # ── DataLoaders ──────────────────────────────────────────────────
    print("\nBuilding meta-episode DataLoaders …")
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

    # ── Model ────────────────────────────────────────────────────────
    model = ContextGoalPredictor(
        num_pois   = 24,
        d_model    = args.d_model,
        nhead      = args.heads,
        num_layers = args.layers,
        dim_ff     = args.dim_ff,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nContextGoalPredictor  —  {n_params:,} trainable parameters")
    print(f"  d_model={args.d_model}, heads={args.heads}, layers={args.layers}, "
          f"dim_ff={args.dim_ff}, K={args.K}, K_recent={args.K_recent}")

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
    history       = {"goal_top1": []}

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss     = train_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        history["goal_top1"].append(val_metrics["goal_top1"])

        is_best = val_metrics["goal_top1"] > best_val_goal
        if is_best:
            best_val_goal = val_metrics["goal_top1"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_metrics": val_metrics},
                best_ckpt,
            )

        epoch_bar.set_postfix(
            tr_loss   = f"{tr_loss:.4f}",
            goal_top1 = f"{val_metrics['goal_top1'] * 100:.1f}%",
            best      = f"{best_val_goal * 100:.1f}%" + (" ✓" if is_best else ""),
        )

    print(f"\nBest val goal top-1: {best_val_goal * 100:.1f}%")
    print(f"Checkpoint → {best_ckpt}")

    # ── Load best checkpoint ─────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ── Goal accuracy at path fractions ──────────────────────────────
    print("\n" + "=" * 57)
    print("  Evaluating goal accuracy at 10% path intervals")
    print("=" * 57)

    split_results: dict[str, dict[float, dict[str, float]]] = {}
    for name, records in tqdm(eval_splits.items(), desc="Evaluating splits", unit="split"):
        split_results[name] = eval_at_fractions(
            model, records, node_coords, poi_to_idx, device,
            K=args.K, K_recent=args.K_recent,
            rng_seed=(args.seed if args.seed is not None else 42),
        )
        tqdm.write(f"\n  {name}  ({len(records)} episodes)")
        tqdm.write(f"  {'Frac':>6}  {'Top-1':>7}  {'Top-5':>7}")
        for frac, vals in split_results[name].items():
            tqdm.write(
                f"  {frac * 100:>5.0f}%  {vals['top1'] * 100:>6.1f}%"
                f"  {vals['top5'] * 100:>6.1f}%"
            )

    # ── Plots: reuse shared helpers from train_bdi_predictor ─────────
    print("\nSaving plots …")
    plot_goal_accuracy(split_results, out_dir)

    # Training curves — goal only
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["goal_top1"], color="#2196F3", linewidth=2, label="Val goal top-1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "accuracy_results.json", "w") as f:
        json.dump(
            {
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
