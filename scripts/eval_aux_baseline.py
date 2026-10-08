"""
auxiliary-supervised unbottlenecked baseline evaluation.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.context_goal_predictor import AuxSupervisedContextGoalPredictor
from model.meta_dataset import MetaEpisodeDataset
from diag_utils import (
    binary_metrics_for_closed,
    get_device,
    load_aux_context,
    load_bdi_bottleneck,
    load_split_records,
    node_coords_from_graph,
    poi_metadata,
    resolve,
    safe_mean,
    write_csv,
    write_text,
)
from scripts.train_bdi_predictor import eval_at_fractions as bdi_eval_at_fractions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auxiliary context baseline evaluation.")
    p.add_argument("--aux-ckpt", default="outputs/diagnostics/sparse/context_goal_aux/seed_0/best.pt")
    p.add_argument("--bnk-ckpt", default="outputs/diagnostics/sparse/bdi_bottleneck/seed_0/best.pt")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--splits", nargs="+", default=["test", "test_new_agent", "test_bh", "test_new_agent_bh"])
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/aux_supervised_baseline")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=48)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _aux_epoch_metrics(model: AuxSupervisedContextGoalPredictor, records, node_coords, poi_to_idx, args, device, split):
    ds = MetaEpisodeDataset(records, node_coords, poi_to_idx, K=args.K, K_recent=args.K_recent)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=MetaEpisodeDataset.collate_fn)
    labels_all, logits_all, belief_t, belief_p, desire_t, desire_logits_all = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            goal_logits, desire_logits, belief_logits = model(
                batch["desire_coords"].to(device),
                batch["desire_mask"].to(device),
                batch["recent_coords"].to(device),
                batch["recent_mask"].to(device),
                batch["current_coords"].to(device),
                batch["current_mask"].to(device),
            )
            labels_all.append(batch["goal_poi_idx"])
            logits_all.append(goal_logits.cpu())
            belief_t.append(batch["belief_vector"])
            belief_p.append(belief_logits.sigmoid().cpu())
            desire_t.append(batch["desire_vector"])
            desire_logits_all.append(desire_logits.cpu())
    labels = torch.cat(labels_all)
    logits = torch.cat(logits_all)
    top5 = logits.topk(5, dim=1).indices
    belief_true = torch.cat(belief_t).numpy().reshape(-1)
    belief_pred = torch.cat(belief_p).numpy().reshape(-1)
    desire_targets = torch.cat(desire_t)
    desire_logits = torch.cat(desire_logits_all)
    bmetrics = binary_metrics_for_closed(belief_true, belief_pred)
    return {
        "goal": {
            "model": "AuxSupervisedContextGoalPredictor",
            "split": split,
            "top1_full": float((top5[:, 0] == labels).float().mean().item()),
            "top5_full": float((top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()),
            "nll_full": float(F.cross_entropy(logits, labels).item()),
        },
        "belief": {"model": "AuxSupervisedContextGoalPredictor", "split": split, **bmetrics},
        "desire": {
            "model": "AuxSupervisedContextGoalPredictor",
            "split": split,
            "desire_kl": float(F.kl_div(F.log_softmax(desire_logits, dim=-1), desire_targets, reduction="batchmean").item()),
        },
    }


def _bdi_epoch_metrics(model, records, node_coords, poi_to_idx, args, device, split):
    ds = MetaEpisodeDataset(records, node_coords, poi_to_idx, K=args.K, K_recent=args.K_recent)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=MetaEpisodeDataset.collate_fn)
    labels_all, logits_all, belief_t, belief_p, desire_t, desire_logits_all = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            goal_logits, desire_logits, belief_logits = model(
                batch["desire_coords"].to(device),
                batch["desire_mask"].to(device),
                batch["recent_coords"].to(device),
                batch["recent_mask"].to(device),
                batch["current_coords"].to(device),
                batch["current_mask"].to(device),
            )
            labels_all.append(batch["goal_poi_idx"])
            logits_all.append(goal_logits.cpu())
            belief_t.append(batch["belief_vector"])
            belief_p.append(belief_logits.sigmoid().cpu())
            desire_t.append(batch["desire_vector"])
            desire_logits_all.append(desire_logits.cpu())
    labels = torch.cat(labels_all)
    logits = torch.cat(logits_all)
    top5 = logits.topk(5, dim=1).indices
    belief_true = torch.cat(belief_t).numpy().reshape(-1)
    belief_pred = torch.cat(belief_p).numpy().reshape(-1)
    desire_targets = torch.cat(desire_t)
    desire_logits = torch.cat(desire_logits_all)
    bmetrics = binary_metrics_for_closed(belief_true, belief_pred)
    return {
        "goal": {
            "model": "BDIBottleneck",
            "split": split,
            "top1_full": float((top5[:, 0] == labels).float().mean().item()),
            "top5_full": float((top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()),
            "nll_full": float(F.cross_entropy(logits, labels).item()),
        },
        "belief": {"model": "BDIBottleneck", "split": split, **bmetrics},
        "desire": {
            "model": "BDIBottleneck",
            "split": split,
            "desire_kl": float(F.kl_div(F.log_softmax(desire_logits, dim=-1), desire_targets, reduction="batchmean").item()),
        },
    }


def _flatten_fraction_results(model_name: str, split: str, results: dict[float, dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {
            "model": model_name,
            "split": split,
            "fraction": frac,
            "top1": vals["top1"],
            "top5": vals["top5"],
        }
        for frac, vals in sorted(results.items())
    ]


def _summary(comparison_rows: list[dict[str, Any]], belief_rows: list[dict[str, Any]]) -> str:
    aux_rows = [r for r in comparison_rows if r["model"] == "AuxSupervisedContextGoalPredictor"]
    bdi_rows = [r for r in comparison_rows if r["model"] == "BDIBottleneck"]
    aux_top1 = safe_mean([float(r.get("top1_full", r.get("top1", 0.0))) for r in aux_rows])
    bdi_top1 = safe_mean([float(r.get("top1_full", r.get("top1", 0.0))) for r in bdi_rows])
    aux_f1 = safe_mean([float(r["closed_f1"]) for r in belief_rows if r["model"] == "AuxSupervisedContextGoalPredictor"])
    bdi_f1 = safe_mean([float(r["closed_f1"]) for r in belief_rows if r["model"] == "BDIBottleneck"])
    return "\n".join(
        [
            "# Auxiliary-Supervised Context Baseline Summary",
            "",
            f"- Mean AuxContext full-path top-1 across evaluated splits: {aux_top1:.4f}.",
            f"- Mean BDIBottleneck full-path top-1 across evaluated splits: {bdi_top1:.4f}.",
            f"- Mean AuxContext closed-class belief F1: {aux_f1:.4f}.",
            f"- Mean BDIBottleneck closed-class belief F1: {bdi_f1:.4f}.",
            "",
            "AuxContext has auxiliary heads but the goal head does not consume the predicted belief/desire channels.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.no_cuda)
    meta = poi_metadata()
    records_by_split = load_split_records(args.splits_dir, args.splits, smoke=args.smoke, smoke_n=args.smoke_n)
    if not records_by_split:
        raise SystemExit(f"No requested split files found under {resolve(args.splits_dir)}")

    aux_path = resolve(args.aux_ckpt)
    bnk_path = resolve(args.bnk_ckpt)
    if not aux_path.exists():
        raise SystemExit(f"Aux checkpoint not found: {aux_path}")
    if not bnk_path.exists():
        raise SystemExit(f"BDIBottleneck checkpoint not found: {bnk_path}")

    aux_model, _ = load_aux_context(aux_path, device)
    bdi_model, _ = load_bdi_bottleneck(bnk_path, device)
    node_coords = node_coords_from_graph()

    goal_fraction_rows: list[dict[str, Any]] = []
    top5_rows: list[dict[str, Any]] = []
    belief_rows: list[dict[str, Any]] = []
    desire_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for split, records in records_by_split.items():
        aux_epoch = _aux_epoch_metrics(aux_model, records, node_coords, meta["poi_to_idx"], args, device, split)
        bdi_epoch = _bdi_epoch_metrics(bdi_model, records, node_coords, meta["poi_to_idx"], args, device, split)
        comparison_rows.extend([aux_epoch["goal"], bdi_epoch["goal"]])
        belief_rows.extend([aux_epoch["belief"], bdi_epoch["belief"]])
        desire_rows.extend([aux_epoch["desire"], bdi_epoch["desire"]])

        aux_frac = bdi_eval_at_fractions(aux_model, records, node_coords, meta["poi_to_idx"], device, K=args.K, K_recent=args.K_recent, batch_size=args.batch, rng_seed=args.seed)
        bdi_frac = bdi_eval_at_fractions(bdi_model, records, node_coords, meta["poi_to_idx"], device, K=args.K, K_recent=args.K_recent, batch_size=args.batch, rng_seed=args.seed)
        rows = _flatten_fraction_results("AuxSupervisedContextGoalPredictor", split, aux_frac)
        rows += _flatten_fraction_results("BDIBottleneck", split, bdi_frac)
        goal_fraction_rows.extend(rows)
        top5_rows.extend(rows)

    write_csv(out_dir / "goal_accuracy_by_fraction.csv", goal_fraction_rows)
    write_csv(out_dir / "top5_accuracy_by_fraction.csv", top5_rows)
    write_csv(out_dir / "belief_metrics.csv", belief_rows)
    write_csv(out_dir / "desire_metrics.csv", desire_rows)
    write_csv(out_dir / "comparison_summary.csv", comparison_rows)
    write_text(out_dir / "summary.md", _summary(comparison_rows, belief_rows))
    print(f"Aux baseline diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
