"""
scripts/train_goal_predictor.py

Train a simple transformer to predict a pedestrian agent's goal POI from
partial trajectories, then evaluate at 10% path-length intervals.

No agent identity, no meta-learning — the model ingests raw (lat, lon)
waypoints and learns to classify the destination across 24 campus POIs.

Usage
-----
  .venv/bin/python scripts/train_goal_predictor.py

Key flags
---------
  --epochs   30         training epochs (default)
  --batch    64         batch size
  --lr       3e-4       learning rate
  --d-model  64         transformer hidden dim
  --layers   3          encoder depth
  --heads    4          attention heads
  --dim-ff   256        feedforward inner dim
  --dropout  0.1        dropout
  --splits   data/splits          input JSONL directory
  --out      outputs/goal_predictor  output directory for checkpoints + plots

Output
------
  <out>/best.pt                     best checkpoint (by val top-1)
  <out>/accuracy_vs_path_fraction.png
  <out>/accuracy_results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.dataset import (
    TrajectoryDataset,
    build_dataloaders,
    build_poi_index,
    node_coords_from_graph,
)
from model.goal_predictor import GoalPredictor
from simulation.episode import EpisodeRecord


# ── Coordinate normalisation (mirrors model/dataset.py _BBOX) ─────────
_BBOX = {
    "lat_min": 32.867,  "lat_max": 32.895,
    "lon_min": -117.253, "lon_max": -117.220,
}

def _norm(lat: float, lon: float) -> tuple[float, float]:
    lat_n = (lat - _BBOX["lat_min"]) / (_BBOX["lat_max"] - _BBOX["lat_min"])
    lon_n = (lon - _BBOX["lon_min"]) / (_BBOX["lon_max"] - _BBOX["lon_min"])
    return lat_n, lon_n


# ── Path-fraction evaluation points ───────────────────────────────────
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ──────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────

def random_prefix(
    path_coords: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each sample in the batch, truncate its path to a uniformly-sampled
    random prefix of its real length.  Teaches the model to predict the goal
    from any point along the trajectory.
    """
    B = path_coords.size(0)
    new_coords = torch.zeros_like(path_coords)
    new_mask   = torch.zeros_like(attention_mask)
    for i in range(B):
        real_len = int(attention_mask[i].sum().item())
        k = torch.randint(1, real_len + 1, ()).item()
        new_coords[i, :k] = path_coords[i, :k]
        new_mask[i,   :k] = 1
    return new_coords, new_mask


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        coords = batch["path_coords"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["goal_poi_idx"].to(device)

        coords, mask = random_prefix(coords, mask)

        logits = model(coords, mask)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def eval_full_path(model, loader, criterion, device) -> tuple[float, float]:
    """Quick full-path loss + top-1 accuracy on a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    with torch.no_grad():
        for batch in loader:
            coords = batch["path_coords"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["goal_poi_idx"].to(device)
            logits = model(coords, mask)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ──────────────────────────────────────────────────────────────────────
# Fraction evaluation
# ──────────────────────────────────────────────────────────────────────

def eval_at_fractions(
    model,
    records:     list[EpisodeRecord],
    node_coords: dict[int, tuple[float, float]],
    poi_to_idx:  dict[str, int],
    device,
    batch_size:  int = 256,
) -> dict[float, dict[str, float]]:
    """
    Evaluate top-1 and top-5 goal-POI accuracy at each fraction in FRACTIONS.

    At fraction f, each episode's path is truncated to the first
    max(1, round(f × T)) waypoints before being fed to the model.

    Returns
    -------
    {fraction: {"top1": float, "top5": float}}
    """
    model.eval()
    results: dict[float, dict[str, float]] = {}

    for frac in tqdm(FRACTIONS, desc="Fractions", unit="frac", leave=False):
        correct1 = 0
        correct5 = 0

        for start in range(0, len(records), batch_size):
            batch_recs = records[start : start + batch_size]

            coords_list: list[torch.Tensor] = []
            labels: list[int] = []

            for r in batch_recs:
                T = len(r.path)
                k = max(1, round(frac * T))
                pts = []
                for nid in r.path[:k]:
                    lat, lon = node_coords.get(nid, (0.0, 0.0))
                    pts.append(_norm(lat, lon))
                coords_list.append(torch.tensor(pts, dtype=torch.float32))
                labels.append(poi_to_idx[r.goal_poi])

            # Dynamic padding within this mini-batch
            max_t = max(c.size(0) for c in coords_list)
            B     = len(coords_list)
            pad_coords = torch.zeros(B, max_t, 2)
            attn_mask  = torch.zeros(B, max_t, dtype=torch.long)
            for i, c in enumerate(coords_list):
                pad_coords[i, : c.size(0)] = c
                attn_mask[i,  : c.size(0)] = 1

            labels_t = torch.tensor(labels)

            with torch.no_grad():
                logits = model(pad_coords.to(device), attn_mask.to(device))

            top5 = logits.topk(5, dim=1).indices.cpu()
            correct1 += (top5[:, 0] == labels_t).sum().item()
            correct5 += (top5 == labels_t.unsqueeze(1)).any(1).sum().item()

        n = len(records)
        results[frac] = {"top1": correct1 / n, "top5": correct5 / n}

    return results


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

_SPLIT_STYLE: dict[str, dict] = {
    "val":            {"color": "#2196F3", "label": "Val"},
    "test":           {"color": "#FF9800", "label": "Test"},
    "test_new_agent": {"color": "#9C27B0", "label": "Test (new agent)"},
}


def plot_accuracy(
    split_results: dict[str, dict[float, dict[str, float]]],
    out_dir: Path,
) -> None:
    """
    Two side-by-side line plots: top-1 and top-5 accuracy vs path fraction.
    One line per split (val, test, test_new_agent).
    """
    pct = [int(f * 100) for f in FRACTIONS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(
        axes,
        ["top1", "top5"],
        ["Top-1 Goal Accuracy", "Top-5 Goal Accuracy"],
    ):
        for split_name, res in split_results.items():
            style = _SPLIT_STYLE.get(split_name, {"color": "gray", "label": split_name})
            ys = [res[f][metric] * 100 for f in FRACTIONS]
            ax.plot(
                pct, ys,
                marker="o", linewidth=2, markersize=5,
                color=style["color"], label=style["label"],
            )

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
    out_path = out_dir / "accuracy_vs_path_fraction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI + main
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GoalPredictor transformer on BDI ToM trajectories."
    )
    p.add_argument("--epochs",  type=int,   default=30,   metavar="N")
    p.add_argument("--batch",   type=int,   default=64,   metavar="N")
    p.add_argument("--lr",      type=float, default=3e-4, metavar="LR")
    p.add_argument("--wd",      type=float, default=1e-4, metavar="WD",
                   help="AdamW weight decay")
    p.add_argument("--d-model", type=int,   default=64,   metavar="D",
                   dest="d_model")
    p.add_argument("--layers",  type=int,   default=3,    metavar="N")
    p.add_argument("--heads",   type=int,   default=4,    metavar="N")
    p.add_argument("--dim-ff",  type=int,   default=256,  metavar="D",
                   dest="dim_ff")
    p.add_argument("--dropout", type=float, default=0.1,  metavar="P")
    p.add_argument("--splits",  type=str,   default="data/splits",
                   metavar="DIR", help="Directory containing *.jsonl splits")
    p.add_argument("--out",     type=str,   default="outputs/goal_predictor",
                   metavar="DIR", help="Output directory")
    p.add_argument("--seed",    type=int,   default=None, metavar="N",
                   help="Random seed for training (if set, writes to out/seed_<N>)")
    p.add_argument("--no-cuda", action="store_true",
                   help="Disable GPU/MPS even if available")
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
    args    = parse_args()
    out_dir = ROOT / args.out
    if args.seed is not None:
        out_dir = out_dir / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = ROOT / args.splits

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

    # ── Configs ──────────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)

    # Build POI → index mapping directly from config (order matches Agent)
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

    # ── DataLoaders for training ──────────────────────────────────────
    print("\nBuilding DataLoaders...")
    loaders = build_dataloaders(
        splits_dir  = splits_dir,
        node_coords = node_coords,
        poi_cfg     = poi_cfg,
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
    model = GoalPredictor(
        num_pois   = 24,
        d_model    = args.d_model,
        nhead      = args.heads,
        num_layers = args.layers,
        dim_ff     = args.dim_ff,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nGoalPredictor  —  {n_params:,} trainable parameters")
    print(f"  d_model={args.d_model}, heads={args.heads}, "
          f"layers={args.layers}, dim_ff={args.dim_ff}, dropout={args.dropout}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_acc = 0.0
    best_ckpt    = out_dir / "best.pt"

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr_loss           = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_full_path(model, val_loader, criterion, device)
        scheduler.step()

        epoch_bar.set_postfix(
            tr_loss  = f"{tr_loss:.4f}",
            val_loss = f"{val_loss:.4f}",
            val_top1 = f"{val_acc * 100:.1f}%",
            best     = f"{best_val_acc * 100:.1f}%",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc},
                best_ckpt,
            )
            epoch_bar.set_postfix(
                tr_loss  = f"{tr_loss:.4f}",
                val_loss = f"{val_loss:.4f}",
                val_top1 = f"{val_acc * 100:.1f}%",
                best     = f"{best_val_acc * 100:.1f}% ✓",
            )

    print(f"\nBest val top-1 (full path): {best_val_acc * 100:.1f}%")
    print(f"Checkpoint saved → {best_ckpt}")

    # ── Load best checkpoint ──────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ── Fraction evaluation ───────────────────────────────────────────
    print("\n" + "=" * 57)
    print("  Evaluating at 10% path intervals")
    print("=" * 57)

    split_results: dict[str, dict[float, dict[str, float]]] = {}

    for name, records in tqdm(eval_splits.items(), desc="Evaluating splits", unit="split"):
        split_results[name] = eval_at_fractions(
            model, records, node_coords, poi_to_idx, device
        )
        # Print per-fraction table after each split
        tqdm.write(f"\n  {name}  ({len(records)} episodes)")
        tqdm.write(f"  {'Frac':>6}  {'Top-1':>7}  {'Top-5':>7}")
        for frac, vals in split_results[name].items():
            tqdm.write(
                f"  {frac * 100:>5.0f}%  {vals['top1'] * 100:>6.1f}%  "
                f"{vals['top5'] * 100:>6.1f}%"
            )

    # ── Plot ──────────────────────────────────────────────────────────
    print("\nSaving plot...")
    plot_accuracy(split_results, out_dir)

    # Save raw results as JSON for later analysis
    results_path = out_dir / "accuracy_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {s: {str(k): v for k, v in r.items()} for s, r in split_results.items()},
            f,
            indent=2,
        )
    print(f"  Results saved → {results_path}")


if __name__ == "__main__":
    main()
