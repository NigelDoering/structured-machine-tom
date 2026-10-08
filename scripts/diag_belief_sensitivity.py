"""
gradient and parameter sensitivity for the belief channel.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from diag_utils import (
    collect_bdi_components,
    get_device,
    load_bdi_bottleneck,
    load_split_records,
    node_coords_from_graph,
    poi_metadata,
    preference_rank,
    resolve,
    safe_mean,
    safe_se,
    write_csv,
    write_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Belief gradient/weight sensitivity.")
    p.add_argument("--bnk-ckpt", default="outputs/diagnostics/sparse/bdi_bottleneck/seed_0/best.pt")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--splits", nargs="+", default=["test", "test_new_agent", "test_bh", "test_new_agent_bh"])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/belief_sensitivity")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=48)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _weight_norm_rows(model, num_pois: int) -> list[dict[str, Any]]:
    first = model.goal_head[0]
    weight = first.weight.detach().cpu()
    segments = {
        "desire": weight[:, :num_pois],
        "belief": weight[:, num_pois : 2 * num_pois],
        "trajectory": weight[:, 2 * num_pois :],
    }
    return [
        {
            "segment": name,
            "l2_norm": float(torch.linalg.vector_norm(values).item()),
            "frobenius_norm": float(torch.linalg.matrix_norm(values).item()),
            "mean_abs_weight": float(values.abs().mean().item()),
            "n_parameters": int(values.numel()),
        }
        for name, values in segments.items()
    ]


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    metrics = [
        "grad_true_goal_abs",
        "grad_same_category_alt_abs",
        "grad_different_category_abs",
        "grad_believed_closed_abs",
        "grad_believed_open_abs",
        "ratio_true_goal_to_different_category",
    ]
    out = []
    for key, vals in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        base = {keys[i]: key[i] for i in range(len(keys))}
        base["n_episodes"] = len(vals)
        for metric in metrics:
            data = [float(v[metric]) for v in vals if not np.isnan(float(v[metric]))]
            base[metric] = safe_mean(data)
            base[f"{metric}_se"] = safe_se(data)
        out.append(base)
    return out


def _summary_text(summary_rows: list[dict[str, Any]], weight_rows: list[dict[str, Any]]) -> str:
    all_rows = [r for r in summary_rows if r.get("group") == "all"]
    true_grad = safe_mean([float(r["grad_true_goal_abs"]) for r in all_rows])
    diff_grad = safe_mean([float(r["grad_different_category_abs"]) for r in all_rows])
    belief_norm = next((r["l2_norm"] for r in weight_rows if r["segment"] == "belief"), float("nan"))
    desire_norm = next((r["l2_norm"] for r in weight_rows if r["segment"] == "desire"), float("nan"))
    trajectory_norm = next((r["l2_norm"] for r in weight_rows if r["segment"] == "trajectory"), float("nan"))
    return "\n".join(
        [
            "# Belief Sensitivity Summary",
            "",
            f"- Mean |d log p(goal) / d b_goal|: {true_grad:.6f}.",
            f"- Mean different-category belief gradient: {diff_grad:.6f}.",
            f"- Goal-head first-layer L2 norms: desire={desire_norm:.6f}, belief={belief_norm:.6f}, trajectory={trajectory_norm:.6f}.",
            "",
            "Gradients are causal-local diagnostics of the trained goal head. Weight norms are supporting diagnostics only.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.fractions = args.fractions[:1]
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.no_cuda)
    meta = poi_metadata()
    records_by_split = load_split_records(args.splits_dir, args.splits, smoke=args.smoke, smoke_n=args.smoke_n)
    if not records_by_split:
        raise SystemExit(f"No requested split files found under {resolve(args.splits_dir)}")
    ckpt_path = resolve(args.bnk_ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    model, _ = load_bdi_bottleneck(ckpt_path, device)
    node_coords = node_coords_from_graph()

    rows: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        for fraction in args.fractions:
            comps_all = collect_bdi_components(
                model,
                records,
                node_coords,
                meta["poi_to_idx"],
                device,
                fraction=fraction,
                k=args.K,
                k_recent=args.K_recent,
                batch_size=args.batch,
                seed=args.seed,
            )
            comps = [
                c
                for c in comps_all
                if preference_rank(c["desire_target"].numpy(), int(c["goal_idx"])) <= args.top_k
            ]
            for start in range(0, len(comps), args.batch):
                batch = comps[start : start + args.batch]
                dp = torch.stack([c["desire_probs"] for c in batch]).to(device)
                bp = torch.stack([c["belief_probs"] for c in batch]).to(device)
                zt = torch.stack([c["z_tau"] for c in batch]).to(device)
                bp = bp.clone().detach().requires_grad_(True)
                labels = torch.tensor([int(c["goal_idx"]) for c in batch], dtype=torch.long, device=device)
                logits = model.predict_goal(dp, bp, zt)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[torch.arange(labels.numel(), device=device), labels].sum()
                if bp.grad is not None:
                    bp.grad.zero_()
                selected.backward()
                grads = bp.grad.detach().abs().cpu()

                for i, comp in enumerate(batch):
                    goal_idx = int(comp["goal_idx"])
                    goal_cat = meta["idx_to_category"][goal_idx]
                    same_alt = [j for j in meta["category_to_indices"][goal_cat] if j != goal_idx]
                    diff = [j for j in range(meta["num_pois"]) if meta["idx_to_category"][j] != goal_cat]
                    target = comp["belief_target"]
                    closed_idx = torch.where(target == 0)[0].tolist()
                    open_idx = torch.where(target == 1)[0].tolist()
                    diff_mean = float(grads[i, diff].mean().item()) if diff else float("nan")
                    true_grad = float(grads[i, goal_idx].item())
                    rows.append(
                        {
                            "split": split,
                            "fraction": fraction,
                            "record_index": comp["record_index"],
                            "agent_id": comp["agent_id"],
                            "character": comp["character"],
                            "config": comp["config"],
                            "category": goal_cat,
                            "goal_poi": comp["goal_poi"],
                            "grad_true_goal_abs": true_grad,
                            "grad_same_category_alt_abs": float(grads[i, same_alt].mean().item()) if same_alt else float("nan"),
                            "grad_different_category_abs": diff_mean,
                            "grad_believed_closed_abs": float(grads[i, closed_idx].mean().item()) if closed_idx else float("nan"),
                            "grad_believed_open_abs": float(grads[i, open_idx].mean().item()) if open_idx else float("nan"),
                            "ratio_true_goal_to_different_category": true_grad / diff_mean if diff_mean and not np.isnan(diff_mean) else float("nan"),
                        }
                    )

    summary = _aggregate(rows, ["split", "fraction"])
    summary_all = _aggregate(rows, [])
    for row in summary_all:
        row["group"] = "all"
    by_config = _aggregate(rows, ["split", "fraction", "config"])
    by_category = _aggregate(rows, ["split", "fraction", "category"])
    weight_rows = _weight_norm_rows(model, meta["num_pois"])

    write_csv(out_dir / "gradient_sensitivity.csv", rows)
    write_csv(out_dir / "gradient_sensitivity_summary.csv", summary_all + summary + by_config + by_category)
    write_csv(out_dir / "goal_head_input_weight_norms.csv", weight_rows)
    write_text(out_dir / "summary.md", _summary_text(summary_all, weight_rows))
    print(f"Belief sensitivity diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
