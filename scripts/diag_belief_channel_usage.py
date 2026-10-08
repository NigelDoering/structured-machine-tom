"""
test whether the BDIBottleneck goal head uses belief.

All variants hold predicted desire and current trajectory embedding fixed, then
replace only the belief channel before calling model.predict_goal().
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
    p = argparse.ArgumentParser(description="Belief-channel usage diagnostics.")
    p.add_argument("--bnk-ckpt", default="outputs/diagnostics/sparse/bdi_bottleneck/seed_0/best.pt")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--splits", nargs="+", default=["test", "test_new_agent", "test_bh", "test_new_agent_bh"])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--shuffle-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/belief_channel_usage")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=48)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _variant_beliefs(
    variant: str,
    belief_probs: torch.Tensor,
    belief_true: torch.Tensor,
    rng: np.random.Generator,
    shuffle_seed: int | None,
) -> torch.Tensor:
    if variant == "normal":
        return belief_probs
    if variant == "all_open":
        return torch.ones_like(belief_probs)
    if variant == "all_closed":
        return torch.zeros_like(belief_probs)
    if variant == "oracle_true":
        return belief_true.float()
    if variant == "random_rate_matched":
        open_rate = float(belief_true.float().mean().item())
        sampled = rng.binomial(1, open_rate, size=tuple(belief_probs.shape))
        return torch.tensor(sampled, dtype=torch.float32)
    if variant == "shuffle":
        if shuffle_seed is None:
            raise ValueError("shuffle variant needs a shuffle seed")
        local_rng = np.random.default_rng(shuffle_seed)
        order = local_rng.permutation(belief_probs.size(0))
        return belief_probs[torch.tensor(order, dtype=torch.long)]
    raise ValueError(f"Unknown variant: {variant}")


def _subset_flags(comp: dict[str, Any], meta: dict[str, Any]) -> dict[str, bool]:
    target = comp["belief_target"].numpy()
    desire_target = comp["desire_target"].numpy()
    goal_idx = int(comp["goal_idx"])
    goal_cat = meta["idx_to_category"][goal_idx]
    same_cat = meta["category_to_indices"][goal_cat]
    desire_cat = str(comp["desire"])
    desire_cat_indices = meta["category_to_indices"].get(desire_cat, [])
    top_pref_idx = int(np.argmax(desire_target))
    return {
        "all": True,
        "true_goal_top3_preference": preference_rank(desire_target, goal_idx) <= 3,
        "desire_category_has_closed_poi": any(target[i] == 0 for i in desire_cat_indices),
        "top_preference_poi_closed": bool(target[top_pref_idx] == 0),
        "heldout_belief_config": str(comp["config"]).startswith("BH"),
        "early_path_fraction": False,
        "late_path_fraction": False,
        "same_category_has_closed_alt": any(target[i] == 0 for i in same_cat if i != goal_idx),
    }


def _episode_metric_rows(
    split: str,
    fraction: float,
    variant: str,
    shuffle_seed: int | None,
    comps: list[dict[str, Any]],
    logits: torch.Tensor,
    normal_probs: torch.Tensor,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    probs = F.softmax(logits, dim=-1)
    labels = torch.tensor([int(c["goal_idx"]) for c in comps], dtype=torch.long)
    eps = 1e-12
    kl = (normal_probs * ((normal_probs + eps).log() - (probs + eps).log())).sum(dim=1)
    mad = (probs - normal_probs).abs().mean(dim=1)
    rows = []
    for i, comp in enumerate(comps):
        goal_idx = int(comp["goal_idx"])
        goal_cat = meta["idx_to_category"][goal_idx]
        same_cat_alts = [j for j in meta["category_to_indices"][goal_cat] if j != goal_idx]
        true_closed = comp["belief_target"] == 0
        top5 = probs[i].topk(min(5, probs.size(1))).indices
        rows.append(
            {
                "split": split,
                "fraction": fraction,
                "variant": variant,
                "shuffle_seed": "" if shuffle_seed is None else shuffle_seed,
                "record_index": comp["record_index"],
                "agent_id": comp["agent_id"],
                "character": comp["character"],
                "config": comp["config"],
                "desire": comp["desire"],
                "goal_poi": comp["goal_poi"],
                "top1_correct": int(top5[0].item() == labels[i].item()),
                "top5_correct": int((top5 == labels[i]).any().item()),
                "nll": float(F.cross_entropy(logits[i : i + 1], labels[i : i + 1]).item()),
                "true_goal_prob": float(probs[i, goal_idx].item()),
                "believed_closed_prob_mass": float(probs[i, true_closed].sum().item()),
                "same_category_alt_prob_mass": float(probs[i, same_cat_alts].sum().item()),
                "kl_from_normal": float(kl[i].item()),
                "mean_abs_change_from_normal": float(mad[i].item()),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, vals in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        agg = {group_keys[i]: key[i] for i in range(len(group_keys))}
        for metric in [
            "top1_correct",
            "top5_correct",
            "nll",
            "true_goal_prob",
            "believed_closed_prob_mass",
            "same_category_alt_prob_mass",
            "kl_from_normal",
            "mean_abs_change_from_normal",
        ]:
            values = [float(v[metric]) for v in vals]
            agg[metric if metric not in {"top1_correct", "top5_correct"} else metric.replace("_correct", "")] = safe_mean(values)
            agg[f"{metric}_se"] = safe_se(values)
        agg["n_episodes"] = len(vals)
        out.append(agg)
    return out


def _summary(overall_rows: list[dict[str, Any]], subset_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Belief Channel Usage Summary",
        "",
        "Each variant holds predicted desire and `z_tau` fixed and changes only the belief channel.",
        "",
    ]
    all_open_rows = [r for r in overall_rows if r.get("variant") == "all_open"]
    oracle_rows = [r for r in overall_rows if r.get("variant") == "oracle_true"]
    shuffle_rows = [r for r in overall_rows if r.get("variant") == "shuffle"]
    if all_open_rows:
        lines.append(
            f"- Mean KL from normal under all-open masking: {safe_mean([float(r['kl_from_normal']) for r in all_open_rows]):.6f}."
        )
    if shuffle_rows:
        lines.append(
            f"- Mean KL from normal under shuffled beliefs: {safe_mean([float(r['kl_from_normal']) for r in shuffle_rows]):.6f}."
        )
    if oracle_rows:
        lines.append(
            f"- Mean oracle true-goal probability: {safe_mean([float(r['true_goal_prob']) for r in oracle_rows]):.6f}."
        )
    relevant = [
        r
        for r in subset_rows
        if r.get("subset") in {"desire_category_has_closed_poi", "top_preference_poi_closed", "same_category_has_closed_alt"}
    ]
    if relevant:
        lines.append(
            f"- Belief-relevant subset rows written: {len(relevant)}. Compare KL and accuracy changes against the `all` subset."
        )
    lines += [
        "",
        "Interpretation guide:",
        "- Tiny all-open/shuffle KL values indicate weak aggregate belief usage.",
        "- Larger effects on belief-relevant subsets indicate conditional belief usage.",
        "- Oracle improvements over predicted belief indicate the belief head is the limiting factor.",
        "- No oracle improvement suggests the goal head or environment does not strongly rely on belief.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.fractions = args.fractions[:1]
        args.shuffle_seeds = args.shuffle_seeds[:1]

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

    episode_rows: list[dict[str, Any]] = []
    subset_rows_raw: list[dict[str, Any]] = []

    for split, records in records_by_split.items():
        for fraction in args.fractions:
            comps = collect_bdi_components(
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
            dp = torch.stack([c["desire_probs"] for c in comps])
            bp = torch.stack([c["belief_probs"] for c in comps])
            zt = torch.stack([c["z_tau"] for c in comps])
            bt = torch.stack([c["belief_target"] for c in comps])
            normal_logits = model.predict_goal(dp.to(device), bp.to(device), zt.to(device)).cpu()
            normal_probs = F.softmax(normal_logits, dim=-1)

            variants: list[tuple[str, int | None]] = [
                ("normal", None),
                ("all_open", None),
                ("all_closed", None),
                ("oracle_true", None),
                ("random_rate_matched", None),
            ]
            variants += [("shuffle", s) for s in args.shuffle_seeds]

            for variant, shuffle_seed in variants:
                rng = np.random.default_rng(args.seed if shuffle_seed is None else shuffle_seed)
                b_variant = _variant_beliefs(variant, bp, bt, rng, shuffle_seed)
                logits = model.predict_goal(dp.to(device), b_variant.to(device), zt.to(device)).cpu()
                rows = _episode_metric_rows(split, fraction, variant, shuffle_seed, comps, logits, normal_probs, meta)
                episode_rows.extend(rows)
                for row, comp in zip(rows, comps):
                    flags = _subset_flags(comp, meta)
                    flags["early_path_fraction"] = fraction <= 0.5
                    flags["late_path_fraction"] = fraction >= 0.75
                    for subset, keep in flags.items():
                        if keep:
                            subset_rows_raw.append({**row, "subset": subset})

    overall = _aggregate([r for r in subset_rows_raw if r["subset"] == "all"], ["split", "variant", "shuffle_seed"])
    by_subset = _aggregate(subset_rows_raw, ["split", "fraction", "subset", "variant", "shuffle_seed"])
    by_fraction = _aggregate([r for r in subset_rows_raw if r["subset"] == "all"], ["split", "fraction", "variant", "shuffle_seed"])
    by_config = _aggregate([r for r in subset_rows_raw if r["subset"] == "all"], ["split", "fraction", "config", "variant", "shuffle_seed"])

    write_csv(out_dir / "usage_metrics_overall.csv", overall)
    write_csv(out_dir / "usage_metrics_by_subset.csv", by_subset)
    write_csv(out_dir / "usage_metrics_by_fraction.csv", by_fraction)
    write_csv(out_dir / "usage_metrics_by_config.csv", by_config)
    write_csv(
        out_dir / "usage_kl_distribution.csv",
        [
            {
                "split": r["split"],
                "fraction": r["fraction"],
                "variant": r["variant"],
                "shuffle_seed": r["shuffle_seed"],
                "record_index": r["record_index"],
                "config": r["config"],
                "kl_from_normal": r["kl_from_normal"],
                "mean_abs_change_from_normal": r["mean_abs_change_from_normal"],
            }
            for r in episode_rows
        ],
    )
    write_csv(out_dir / "usage_episode_level.csv", episode_rows)
    write_text(out_dir / "summary.md", _summary(overall, by_subset))

    print(f"Belief channel usage diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
