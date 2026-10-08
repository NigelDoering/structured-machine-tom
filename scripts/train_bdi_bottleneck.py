"""
scripts/train_bdi_bottleneck.py

Train BDIBottleneckPredictor: the bottleneck variant of BDIPredictor in which
the goal head receives cat[desire_probs, belief_probs, z_tau] instead of
cat[z_d, z_r, z_tau].  See model/bdi_bottleneck.py for motivation.

All training logic (loss computation, evaluation, plotting) is shared with
train_bdi_predictor.py — only the model class and output directory differ.

Usage
-----
  .venv/bin/python scripts/train_bdi_bottleneck.py
  .venv/bin/python scripts/train_bdi_bottleneck.py --epochs 30 --out outputs/bdi_bottleneck

Key flags  (identical to train_bdi_predictor.py)
---------
  --epochs   30
  --batch    32
  --K        10
  --K-recent  5
  --lr       3e-4
  --lam-g    1.0
  --lam-d    1.0
  --lam-b    5.0
  --d-model  64
  --layers   3
  --out      outputs/bdi_bottleneck

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
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np

from model.bdi_bottleneck import BDIBottleneckPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.meta_dataset import build_meta_dataloaders
from simulation.episode import EpisodeRecord

# ── Re-use all shared training utilities from train_bdi_predictor ─────
from scripts.train_bdi_predictor import (
    BELIEF_LOSS_CHOICES,
    FRACTIONS,
    random_prefix,
    train_epoch,
    eval_epoch,
    eval_at_fractions,
    plot_goal_accuracy,
    plot_training_curves,
    _load_records,
)
from scripts.diag_utils import adamw_kwargs, maybe_force_math_sdp


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BDIBottleneckPredictor (bottleneck goal-head variant)."
    )
    p.add_argument("--epochs",  type=int,   default=30,   metavar="N")
    p.add_argument("--batch",   type=int,   default=32,   metavar="N")
    p.add_argument("--K",        type=int,   default=10,   metavar="K")
    p.add_argument("--K-recent", type=int,   default=5,    metavar="K", dest="K_recent")
    p.add_argument("--lr",      type=float, default=3e-4, metavar="LR")
    p.add_argument("--wd",      type=float, default=1e-4, metavar="WD")
    p.add_argument("--lam-g",   type=float, default=1.0,  metavar="W", dest="lam_g")
    p.add_argument("--lam-d",   type=float, default=1.0,  metavar="W", dest="lam_d")
    p.add_argument("--lam-b",   type=float, default=5.0,  metavar="W", dest="lam_b")
    p.add_argument("--belief-weight", type=float, default=5.0, metavar="W",
                   dest="belief_weight")
    p.add_argument("--belief-loss", choices=BELIEF_LOSS_CHOICES, default="weighted_bce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--balanced-max-weight", type=float, default=12.0)
    p.add_argument("--d-model", type=int,   default=64,   metavar="D", dest="d_model")
    p.add_argument("--layers",  type=int,   default=3,    metavar="N")
    p.add_argument("--heads",   type=int,   default=4,    metavar="N")
    p.add_argument("--dim-ff",  type=int,   default=256,  metavar="D", dest="dim_ff")
    p.add_argument("--warmup",  type=int,   default=0,    metavar="N")
    p.add_argument("--dropout", type=float, default=0.1,  metavar="P")
    p.add_argument("--splits",  type=str,   default="data/splits",          metavar="DIR")
    p.add_argument("--out",     type=str,   default="outputs/bdi_bottleneck", metavar="DIR")
    p.add_argument("--seed",    type=int,   default=None, metavar="N",
                   help="Random seed for training (if set, writes to out/seed_<N>)")
    p.add_argument("--save-every", type=int, default=0, metavar="N",
                   dest="save_every",
                   help="Optionally save epoch_NNN.pt every N epochs for diagnostics (default: disabled)")
    p.add_argument("--resume-state", default="", help="Optional resumable training state checkpoint.")
    p.add_argument("--state-out", default="", help="Path for automatic last-state checkpoint after each epoch.")
    p.add_argument("--max-epochs-this-run", type=int, default=0, help="Optional per-process epoch cap; 0 means run to --epochs.")
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _resolve_optional(path: str, default: Path) -> Path:
    if not path:
        return default
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _state_payload(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: dict[str, list[float]],
    best_val_goal: float,
    best_ckpt: Path,
    model_kwargs: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "completed_epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "best_val_goal": best_val_goal,
        "best_ckpt": str(best_ckpt),
        "model_kwargs": model_kwargs,
        "args": vars(args),
    }


def _save_training_state(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: dict[str, list[float]],
    best_val_goal: float,
    best_ckpt: Path,
    model_kwargs: dict[str, object],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _state_payload(
            epoch,
            model,
            optimizer,
            scheduler,
            history,
            best_val_goal,
            best_ckpt,
            model_kwargs,
            args,
        ),
        path,
    )


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

    # ── Configs ──────────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order  = [poi["id"] for cat_data in poi_cfg["categories"].values()
                  for poi in cat_data["pois"]]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    # ── Graph ─────────────────────────────────────────────────────────
    import osmnx as ox
    graph_path = ROOT / "data" / "ucsd_walk.graphml"
    print(f"Loading graph from {graph_path}")
    G           = ox.load_graphml(graph_path)
    node_coords = node_coords_from_graph(G)
    print(f"Graph loaded  ({len(node_coords):,} nodes)")

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
    model_kwargs = {
        "num_pois": 24,
        "d_model": args.d_model,
        "nhead": args.heads,
        "num_layers": args.layers,
        "dim_ff": args.dim_ff,
        "dropout": args.dropout,
    }
    model = BDIBottleneckPredictor(**model_kwargs).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBDIBottleneckPredictor  —  {n_params:,} trainable parameters")
    print(f"  d_model={args.d_model}, heads={args.heads}, layers={args.layers}, "
          f"dim_ff={args.dim_ff}, K={args.K}, K_recent={args.K_recent}")
    print(f"  λ_goal={args.lam_g}, λ_desire={args.lam_d}, λ_belief={args.lam_b}, "
          f"belief_loss={args.belief_loss}, belief_weight={args.belief_weight}")

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
    history       = {"belief_acc": [], "belief_f1": [], "desire_kl": [], "goal_top1": []}
    state_out = _resolve_optional(args.state_out, out_dir / "last_state.pt")
    completed_epoch = 0

    resume_state = _resolve_optional(args.resume_state, out_dir / "last_state.pt") if args.resume_state else None
    if resume_state is not None:
        if resume_state.exists():
            state = torch.load(resume_state, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            history = {
                key: [float(value) for value in state.get("history", {}).get(key, [])]
                for key in history
            }
            best_val_goal = float(state.get("best_val_goal", best_val_goal))
            best_ckpt = _resolve_optional(str(state.get("best_ckpt", best_ckpt)), best_ckpt)
            completed_epoch = int(state.get("completed_epoch", 0))
            print(f"Resumed training state from {resume_state} at epoch {completed_epoch}")
        else:
            print(f"Resume state not found, starting fresh: {resume_state}")

    if completed_epoch >= args.epochs:
        print(f"Training already complete at epoch {completed_epoch} / {args.epochs}")
    max_epochs_this_run = args.max_epochs_this_run if args.max_epochs_this_run > 0 else args.epochs
    run_stop_epoch = min(args.epochs, completed_epoch + max_epochs_this_run)

    epoch_bar = tqdm(range(completed_epoch + 1, run_stop_epoch + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss     = train_epoch(
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
        ckpt_obj = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        if is_best:
            best_val_goal = val_metrics["goal_top1"]
            torch.save(ckpt_obj, best_ckpt)
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt_obj, out_dir / f"epoch_{epoch:03d}.pt")
        _save_training_state(
            state_out,
            epoch,
            model,
            optimizer,
            scheduler,
            history,
            best_val_goal,
            best_ckpt,
            model_kwargs,
            args,
        )
        completed_epoch = epoch

        epoch_bar.set_postfix(
            tr_loss    = f"{tr_loss:.4f}",
            goal_top1  = f"{val_metrics['goal_top1'] * 100:.1f}%",
            belief_f1  = f"{val_metrics['belief_f1'] * 100:.1f}%",
            best       = f"{best_val_goal * 100:.1f}%" + (" ✓" if is_best else ""),
        )

    print(f"\nBest val goal top-1: {best_val_goal * 100:.1f}%")
    print(f"Checkpoint → {best_ckpt}")

    if completed_epoch < args.epochs:
        print(f"Chunk complete at epoch {completed_epoch} / {args.epochs}; state → {state_out}")
        return

    # ── Load best checkpoint ──────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # ── Goal accuracy at path intervals ───────────────────────────────
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

    # ── Plots & results ───────────────────────────────────────────────
    print("\nSaving plots...")
    plot_goal_accuracy(split_results, out_dir)
    plot_training_curves(history, out_dir)

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
