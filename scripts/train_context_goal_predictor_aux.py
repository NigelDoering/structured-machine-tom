"""
Train AuxSupervisedContextGoalPredictor.

This baseline matches ContextGoalPredictor's unbottlenecked goal path
cat[z_d, z_r, z_tau] while adding the same auxiliary desire/belief heads and
losses used by BDIBottleneckPredictor.
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

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model.context_goal_predictor import AuxSupervisedContextGoalPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.meta_dataset import build_meta_dataloaders
from simulation.episode import EpisodeRecord
from scripts.train_bdi_predictor import (
    BELIEF_LOSS_CHOICES,
    _load_records,
    eval_at_fractions,
    eval_epoch,
    plot_goal_accuracy,
    plot_training_curves,
    train_epoch,
)
from scripts.diag_utils import adamw_kwargs, maybe_force_math_sdp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train auxiliary-supervised unbottlenecked context baseline.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--lam-g", type=float, default=1.0, dest="lam_g")
    p.add_argument("--lam-d", type=float, default=1.0, dest="lam_d")
    p.add_argument("--lam-b", type=float, default=5.0, dest="lam_b")
    p.add_argument("--belief-weight", type=float, default=5.0, dest="belief_weight")
    p.add_argument("--belief-loss", choices=BELIEF_LOSS_CHOICES, default="weighted_bce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--balanced-max-weight", type=float, default=12.0)
    p.add_argument("--d-model", type=int, default=64, dest="d_model")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=256, dest="dim_ff")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--splits", default="data/splits")
    p.add_argument("--out", default="outputs/diagnostics/sparse/context_goal_aux")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-every", type=int, default=0, dest="save_every")
    p.add_argument("--resume-state", default="", help="Optional resumable training state checkpoint.")
    p.add_argument("--state-out", default="", help="Path for automatic last-state checkpoint after each epoch.")
    p.add_argument("--max-epochs-this-run", type=int, default=0, help="Optional per-process epoch cap; 0 means run to --epochs.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _device(no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.batch = min(args.batch, 8)
        args.K = min(args.K, 2)
        args.K_recent = min(args.K_recent, 2)

    out_dir = ROOT / args.out
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

    device = _device(args.no_cuda)
    print(f"Device : {device}")
    if device.type == "cuda" and maybe_force_math_sdp():
        print("BDI_FORCE_MATH_SDP=1: using math SDPA backend")

    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order = [
        poi["id"]
        for cat_data in poi_cfg["categories"].values()
        for poi in cat_data["pois"]
    ]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    import osmnx as ox

    graph_path = ROOT / "data" / "ucsd_walk.graphml"
    print(f"Loading graph from {graph_path}")
    G = ox.load_graphml(graph_path)
    node_coords = node_coords_from_graph(G)
    print(f"Graph loaded  ({len(node_coords):,} nodes)")

    loaders = build_meta_dataloaders(
        splits_dir=splits_dir,
        node_coords=node_coords,
        poi_to_idx=poi_to_idx,
        K=args.K,
        K_recent=args.K_recent,
        batch_size=args.batch,
        splits=["train", "val"],
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    eval_splits: dict[str, list[EpisodeRecord]] = {}
    for name in ("val", "test", "test_new_agent", "test_bh", "test_new_agent_bh"):
        path = splits_dir / f"{name}.jsonl"
        if path.exists():
            eval_splits[name] = _load_records(path)
            print(f"  Loaded {len(eval_splits[name]):>5} records  <- {name}")

    model_kwargs = {
        "num_pois": 24,
        "d_model": args.d_model,
        "nhead": args.heads,
        "num_layers": args.layers,
        "dim_ff": args.dim_ff,
        "dropout": args.dropout,
    }
    model = AuxSupervisedContextGoalPredictor(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAuxSupervisedContextGoalPredictor - {n_params:,} trainable parameters")
    print(
        f"  lambda_goal={args.lam_g}, lambda_desire={args.lam_d}, "
        f"lambda_belief={args.lam_b}, belief_loss={args.belief_loss}, "
        f"belief_weight={args.belief_weight}"
    )

    opt_kwargs = adamw_kwargs()
    if opt_kwargs.get("foreach") is False:
        print("BDI_DISABLE_FOREACH_OPT=1: disabling AdamW foreach kernels")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, **opt_kwargs)
    if args.warmup > 0:
        warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup), eta_min=args.lr / 10)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    warnings.filterwarnings(
        "ignore",
        message="The epoch parameter in `scheduler.step\\(\\)`",
        category=UserWarning,
    )

    best_val_goal = -1.0
    best_ckpt = out_dir / "best.pt"
    history = {"belief_acc": [], "belief_f1": [], "desire_kl": [], "goal_top1": []}
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

    for epoch in tqdm(range(completed_epoch + 1, run_stop_epoch + 1), desc="Training", unit="epoch"):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args.lam_g,
            args.lam_d,
            args.lam_b,
            args.belief_weight,
            device,
            args.belief_loss,
            args.focal_gamma,
            args.balanced_max_weight,
        )
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        for key in history:
            history[key].append(val_metrics[key])

        is_best = val_metrics["goal_top1"] > best_val_goal
        ckpt_obj = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "val_metrics": val_metrics,
            "train_loss": train_loss,
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

    print(f"\nBest val goal top-1: {best_val_goal * 100:.1f}%")
    print(f"Checkpoint -> {best_ckpt}")

    if completed_epoch < args.epochs:
        print(f"Chunk complete at epoch {completed_epoch} / {args.epochs}; state -> {state_out}")
        return

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    split_results: dict[str, dict[float, dict[str, float]]] = {}
    for name, records in tqdm(eval_splits.items(), desc="Evaluating splits", unit="split"):
        split_results[name] = eval_at_fractions(
            model,
            records,
            node_coords,
            poi_to_idx,
            device,
            K=args.K,
            K_recent=args.K_recent,
            rng_seed=(args.seed if args.seed is not None else 42),
        )

    plot_goal_accuracy(split_results, out_dir)
    plot_training_curves(history, out_dir)
    with open(out_dir / "accuracy_results.json", "w") as f:
        json.dump(
            {
                "goal_fractions": {
                    split: {str(frac): vals for frac, vals in res.items()}
                    for split, res in split_results.items()
                },
                "training_history": history,
                "model": "AuxSupervisedContextGoalPredictor",
            },
            f,
            indent=2,
        )
    print(f"Results -> {out_dir / 'accuracy_results.json'}")


if __name__ == "__main__":
    main()
