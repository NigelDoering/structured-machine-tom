"""
scripts/eval_ablations.py

Compare BDIBottleneckPredictor against its two ablation variants on goal inference.

Loads best.pt checkpoints for all three models and evaluates top-1/top-5 goal
accuracy at each path fraction on test and test_new_agent splits.  Produces:

  results/ablation/top1_test.png            — top-1 accuracy, test split
  results/ablation/top1_new_agent.png       — top-1 accuracy, new-agent split
  results/ablation/top1_combined.png        — 1×2 panel (test | new-agent)
  results/ablation/ablation_results.json    — full numeric results

Usage
-----
  .venv/bin/python scripts/eval_ablations.py \\
      --bdi       outputs/bdi_bottleneck \\
      --no-desire outputs/ablation_no_desire \\
      --no-belief outputs/ablation_no_belief

All three checkpoint paths are optional; a model is skipped if its path is
not supplied or if best.pt does not exist there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from model.bdi_bottleneck import BDIBottleneckPredictor
from model.bdi_bottleneck_ablations import BDIBottleneckNoDesire, BDIBottleneckNoBelief
from model.dataset import build_poi_index, node_coords_from_graph
from simulation.episode import EpisodeRecord
from scripts.train_bdi_predictor import (
    FRACTIONS,
    _load_records,
    _build_agent_index,
    _build_context_tensor,
    _norm,
)
from scripts.train_bdi_bottleneck_ablation import eval_at_fractions_ablation


# ── Model registry ─────────────────────────────────────────────────────────

_MODEL_LABELS = {
    "bdi":        "BDIBottleneck (full)",
    "no_desire":  "No-Desire ablation",
    "no_belief":  "No-Belief ablation",
}

_COLORS = {
    "bdi":        "#4CAF50",   # green — matches eval_goal_inference.py
    "no_desire":  "#FF9800",   # orange
    "no_belief":  "#9C27B0",   # purple
}

_LINESTYLES = {
    "bdi":       "-",
    "no_desire": "--",
    "no_belief": ":",
}


# ── Fraction evaluation for full BDIBottleneck ─────────────────────────────


def eval_at_fractions_full(
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
    """Evaluate full BDIBottleneck at path fractions."""
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

            desire_ctx_list = []
            recent_ctx_list = []
            curr_list       = []
            labels          = []

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
                desire_ctx_list.append(
                    _build_context_tensor([records[i] for i in d_src], node_coords, K)
                )

                preceding = agent_idxs[:pos]
                if len(preceding) == 0:
                    fallback = agent_idxs[pos + 1:] or [flat_idx]
                    r_src = (fallback * ((K_recent // len(fallback)) + 1))[:K_recent]
                elif len(preceding) < K_recent:
                    r_src = (preceding * ((K_recent // len(preceding)) + 1))[:K_recent]
                else:
                    r_src = preceding[-K_recent:]
                recent_ctx_list.append(
                    _build_context_tensor([records[i] for i in r_src], node_coords, K_recent)
                )

                T = len(r.path)
                k = max(1, round(frac * T))
                pts = [
                    _norm(*node_coords.get(nid, (0.0, 0.0)))
                    for nid in r.path[:k]
                ]
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


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_ablation_comparison(
    model_results: dict[str, dict[str, dict[float, dict[str, float]]]],
    out_dir: Path,
) -> None:
    """
    Produce top-1 goal accuracy comparison plots.

    model_results: {model_key: {split_name: {frac: {top1, top5}}}}
    """
    pct = [int(f * 100) for f in FRACTIONS]

    for split in ("test", "test_new_agent"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, split_map in model_results.items():
            if split not in split_map:
                continue
            ys   = [split_map[split][f]["top1"] * 100 for f in FRACTIONS]
            stds = [split_map[split][f].get("top1_std", 0.0) * 100 for f in FRACTIONS]
            ax.plot(
                pct, ys,
                marker="o", linewidth=2, markersize=5,
                color=_COLORS[key],
                linestyle=_LINESTYLES[key],
                label=_MODEL_LABELS[key],
            )
            if any(s > 0 for s in stds):
                ax.fill_between(
                    pct,
                    [y - s for y, s in zip(ys, stds)],
                    [y + s for y, s in zip(ys, stds)],
                    alpha=0.15, color=_COLORS[key],
                )
        split_label = "Test" if split == "test" else "Test (new agent)"
        ax.set_xlabel("Path revealed (%)", fontsize=11)
        ax.set_ylabel("Top-1 accuracy (%)", fontsize=11)
        ax.set_title(f"Ablation comparison — {split_label}", fontsize=12, fontweight="bold")
        ax.set_xlim(5, 105)
        ax.set_ylim(0, 105)
        ax.set_xticks(pct)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9.5)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=2.0)

        name = "top1_test" if split == "test" else "top1_new_agent"
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    # Combined 1×2 panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, split, title in zip(
        axes,
        ["test", "test_new_agent"],
        ["Test", "Test (new agent)"],
    ):
        for key, split_map in model_results.items():
            if split not in split_map:
                continue
            ys   = [split_map[split][f]["top1"] * 100 for f in FRACTIONS]
            stds = [split_map[split][f].get("top1_std", 0.0) * 100 for f in FRACTIONS]
            ax.plot(
                pct, ys,
                marker="o", linewidth=2, markersize=5,
                color=_COLORS[key],
                linestyle=_LINESTYLES[key],
                label=_MODEL_LABELS[key],
            )
            if any(s > 0 for s in stds):
                ax.fill_between(
                    pct,
                    [y - s for y, s in zip(ys, stds)],
                    [y + s for y, s in zip(ys, stds)],
                    alpha=0.15, color=_COLORS[key],
                )
        ax.set_xlabel("Path revealed (%)", fontsize=11)
        ax.set_ylabel("Top-1 accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(5, 105)
        ax.set_ylim(0, 105)
        ax.set_xticks(pct)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9.5)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    path = out_dir / "top1_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def print_summary_table(
    model_results: dict[str, dict[str, dict[float, dict[str, float]]]],
) -> None:
    key_fracs = [0.25, 0.50, 0.75, 1.00]
    splits    = ["test", "test_new_agent"]

    header_parts = [f"{'Model':<30}"]
    for split in splits:
        slabel = "Test" if split == "test" else "New-agent"
        for f in key_fracs:
            header_parts.append(f"{slabel} {int(f*100)}%".rjust(12))
    print("\n" + "  ".join(header_parts))
    print("-" * (30 + 14 * len(splits) * len(key_fracs)))

    for key, split_map in model_results.items():
        row = [f"{_MODEL_LABELS[key]:<30}"]
        for split in splits:
            for f in key_fracs:
                if split in split_map and f in split_map[split]:
                    v = split_map[split][f]["top1"] * 100
                    row.append(f"{v:>11.2f}%")
                else:
                    row.append(f"{'N/A':>12}")
        print("  ".join(row))


# ── Multi-seed aggregation ─────────────────────────────────────────────────


def _aggregate_seeds(
    per_seed: list[dict[str, dict[float, dict[str, float]]]]
) -> dict[str, dict[float, dict[str, float]]]:
    """Average top1/top5 across seeds; add _std entries. n=1 gives std=0 (no band)."""
    if not per_seed:
        return {}
    agg: dict[str, dict[float, dict[str, float]]] = {}
    for split in per_seed[0]:
        agg[split] = {}
        for frac in per_seed[0][split]:
            top1s = [s[split][frac]["top1"] for s in per_seed]
            top5s = [s[split][frac]["top5"] for s in per_seed]
            agg[split][frac] = {
                "top1":     float(np.mean(top1s)),
                "top5":     float(np.mean(top5s)),
                "top1_std": float(np.std(top1s, ddof=0)),
                "top5_std": float(np.std(top5s, ddof=0)),
            }
    return agg


# ── CLI + main ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BDIBottleneck ablation comparison.")
    p.add_argument("--bdi",       type=str, nargs="+",
                   default=["outputs/bdi_bottleneck"],
                   metavar="DIR", help="Checkpoint dir(s) for full BDIBottleneck (one per seed)")
    p.add_argument("--no-desire", type=str, nargs="+",
                   default=["outputs/ablation_no_desire"],
                   metavar="DIR", dest="no_desire",
                   help="Checkpoint dir(s) for BDIBottleneckNoDesire (one per seed)")
    p.add_argument("--no-belief", type=str, nargs="+",
                   default=["outputs/ablation_no_belief"],
                   metavar="DIR", dest="no_belief",
                   help="Checkpoint dir(s) for BDIBottleneckNoBelief (one per seed)")
    p.add_argument("--splits",    type=str, default="data/splits",     metavar="DIR")
    p.add_argument("--out",       type=str, default="results/ablation", metavar="DIR")
    p.add_argument("--K",         type=int, default=10)
    p.add_argument("--K-recent",  type=int, default=5,  dest="K_recent")
    p.add_argument("--no-cuda",   action="store_true")
    return p.parse_args()


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
    print(f"Device: {device}")

    # ── Config & graph ────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order  = [poi["id"] for cat_data in poi_cfg["categories"].values()
                  for poi in cat_data["pois"]]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    import osmnx as ox
    G           = ox.load_graphml(ROOT / "data" / "ucsd_walk.graphml")
    node_coords = node_coords_from_graph(G)
    print(f"Graph loaded  ({len(node_coords):,} nodes)")

    # ── Load evaluation records ───────────────────────────────────────
    eval_splits: dict[str, list[EpisodeRecord]] = {}
    for name in ("test", "test_new_agent"):
        p = splits_dir / f"{name}.jsonl"
        if p.exists():
            eval_splits[name] = _load_records(p)
            print(f"  Loaded {len(eval_splits[name]):>5} records  ← {name}")

    # ── Load checkpoints & evaluate (multi-seed) ─────────────────────
    ckpt_dir_lists = {
        "bdi":       [ROOT / d for d in args.bdi],
        "no_desire": [ROOT / d for d in args.no_desire],
        "no_belief": [ROOT / d for d in args.no_belief],
    }

    model_results: dict[str, dict[str, dict[float, dict[str, float]]]] = {}
    per_seed_store: dict[str, list] = {"bdi": [], "no_desire": [], "no_belief": []}

    for key, dir_list in ckpt_dir_lists.items():
        for seed_i, ckpt_dir in enumerate(dir_list):
            ckpt_path = ckpt_dir / "best.pt"
            if not ckpt_path.exists():
                print(f"\n[skip] {key} seed {seed_i}: {ckpt_path} not found")
                continue

            print(f"\nLoading {key} (seed {seed_i}) from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            mkw = ckpt.get("model_kwargs", {
                "num_pois": 24, "d_model": 64, "nhead": 4,
                "num_layers": 3, "dim_ff": 256, "dropout": 0.1,
            })
            if key == "bdi":
                model = BDIBottleneckPredictor(**mkw).to(device)
            elif key == "no_desire":
                model = BDIBottleneckNoDesire(**mkw).to(device)
            else:
                model = BDIBottleneckNoBelief(**mkw).to(device)

            model.load_state_dict(ckpt["model_state"])
            model.eval()

            seed_res: dict[str, dict[float, dict[str, float]]] = {}
            for name, records in tqdm(
                eval_splits.items(), desc=f"  {key}[{seed_i}]", unit="split"
            ):
                if key == "bdi":
                    seed_res[name] = eval_at_fractions_full(
                        model, records, node_coords, poi_to_idx, device,
                        K=args.K, K_recent=args.K_recent,
                    )
                else:
                    ablation = "no_desire" if key == "no_desire" else "no_belief"
                    seed_res[name] = eval_at_fractions_ablation(
                        model, records, node_coords, poi_to_idx, ablation, device,
                        K=args.K, K_recent=args.K_recent,
                    )
            per_seed_store[key].append(seed_res)

        if per_seed_store[key]:
            n = len(per_seed_store[key])
            print(f"\n  Aggregating {n} seed(s) for {key}")
            model_results[key] = _aggregate_seeds(per_seed_store[key])

    if not model_results:
        print("No checkpoints found. Train models first.")
        return

    # ── Print table ───────────────────────────────────────────────────
    print_summary_table(model_results)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nSaving plots...")
    plot_ablation_comparison(model_results, out_dir)

    # ── Save results JSON ─────────────────────────────────────────────
    out_json = out_dir / "ablation_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                k: {
                    split: {str(frac): vals for frac, vals in fmap.items()}
                    for split, fmap in smap.items()
                }
                for k, smap in model_results.items()
            },
            f, indent=2,
        )
    print(f"  Results → {out_json}")


if __name__ == "__main__":
    main()
