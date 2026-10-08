"""
belief-bit dose response for the true goal.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    p = argparse.ArgumentParser(description="Belief-bit dose response.")
    p.add_argument("--bnk-ckpt", default="outputs/diagnostics/sparse/bdi_bottleneck/seed_0/best.pt")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--splits", nargs="+", default=["test", "test_new_agent", "test_bh", "test_new_agent_bh"])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--doses", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--open-same-category-alts", action="store_true", default=True)
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/belief_dose_response")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=48)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(np.asarray(x, dtype=np.float64))
    ry = _rankdata(np.asarray(y, dtype=np.float64))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, vals in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        base = {keys[i]: key[i] for i in range(len(keys))}
        probs = [float(v["p_goal"]) for v in vals]
        deltas = [float(v["delta_vs_dose_1"]) for v in vals]
        out.append(
            {
                **base,
                "n": len(vals),
                "mean_p_goal": safe_mean(probs),
                "se_p_goal": safe_se(probs),
                "mean_delta_vs_dose_1": safe_mean(deltas),
                "se_delta_vs_dose_1": safe_se(deltas),
            }
        )
    return out


def _episode_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["fraction"], row["record_index"])].append(row)
    out = []
    for (split, fraction, record_index), vals in sorted(grouped.items()):
        vals = sorted(vals, key=lambda r: float(r["dose"]), reverse=True)
        doses = [float(v["dose"]) for v in vals]
        probs = [float(v["p_goal"]) for v in vals]
        mono = all(probs[i] >= probs[i + 1] - 1e-12 for i in range(len(probs) - 1))
        slope = float(np.polyfit(doses, probs, 1)[0]) if len(doses) >= 2 else float("nan")
        out.append(
            {
                "split": split,
                "fraction": fraction,
                "record_index": record_index,
                "config": vals[0]["config"],
                "category": vals[0]["category"],
                "goal_poi": vals[0]["goal_poi"],
                "monotonic_decrease_as_belief_decreases": int(mono),
                "spearman_dose_p_goal": _spearman(doses, probs),
                "linear_slope": slope,
            }
        )
    return out


def _plot(summary: list[dict[str, Any]], out_path) -> None:
    splits = sorted({str(r["split"]) for r in summary})
    if not splits:
        return
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.5), squeeze=False)
    for ax, split in zip(axes[0], splits):
        rows = [r for r in summary if r["split"] == split]
        fractions = sorted({float(r["fraction"]) for r in rows})
        for frac in fractions:
            fr = sorted([r for r in rows if float(r["fraction"]) == frac], key=lambda r: float(r["dose"]))
            xs = [float(r["dose"]) for r in fr]
            ys = [float(r["mean_p_goal"]) for r in fr]
            ses = [float(r["se_p_goal"]) for r in fr]
            ax.errorbar(xs, ys, yerr=ses, marker="o", linewidth=1.8, capsize=3, label=f"{int(frac*100)}%")
        ax.set_title(split)
        ax.set_xlabel("Belief value assigned to true-goal bit")
        ax.set_ylabel("Predicted P(true goal)")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Path")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary_text(episode_summary: list[dict[str, Any]], curve_summary: list[dict[str, Any]]) -> str:
    mono_rate = safe_mean([float(r["monotonic_decrease_as_belief_decreases"]) for r in episode_summary])
    slope = safe_mean([float(r["linear_slope"]) for r in episode_summary])
    return "\n".join(
        [
            "# Belief Dose Response Summary",
            "",
            f"- Eligible episodes: {len(episode_summary)}.",
            f"- Monotonicity rate: {mono_rate:.4f}.",
            f"- Mean per-episode linear slope of P(goal) vs belief dose: {slope:.6f}.",
            f"- Curve rows written: {len(curve_summary)}.",
            "",
            "A monotonic downward curve as the true-goal belief bit moves from 1 to 0 supports functional use of the belief slot.",
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
            if not comps:
                continue
            dp = torch.stack([c["desire_probs"] for c in comps])
            bp_base = torch.stack([c["belief_probs"] for c in comps])
            zt = torch.stack([c["z_tau"] for c in comps])
            labels = torch.tensor([int(c["goal_idx"]) for c in comps], dtype=torch.long)
            p_at_one: dict[int, float] = {}
            for dose in args.doses:
                bp = bp_base.clone()
                for i, comp in enumerate(comps):
                    goal_idx = int(comp["goal_idx"])
                    goal_cat = meta["idx_to_category"][goal_idx]
                    bp[i, goal_idx] = float(dose)
                    if args.open_same_category_alts:
                        for alt_idx in meta["category_to_indices"][goal_cat]:
                            if alt_idx != goal_idx:
                                bp[i, alt_idx] = 1.0
                with torch.no_grad():
                    logits = model.predict_goal(dp.to(device), bp.to(device), zt.to(device)).cpu()
                probs = F.softmax(logits, dim=-1)
                for i, comp in enumerate(comps):
                    p_goal = float(probs[i, labels[i]].item())
                    if abs(dose - 1.0) < 1e-12:
                        p_at_one[i] = p_goal
                    rows.append(
                        {
                            "split": split,
                            "fraction": fraction,
                            "record_index": comp["record_index"],
                            "agent_id": comp["agent_id"],
                            "character": comp["character"],
                            "config": comp["config"],
                            "category": meta["idx_to_category"][int(comp["goal_idx"])],
                            "goal_poi": comp["goal_poi"],
                            "dose": dose,
                            "p_goal": p_goal,
                            "delta_vs_dose_1": p_goal - p_at_one.get(i, p_goal),
                        }
                    )

    # Fill deltas robustly even if dose order was changed.
    one_lookup = {
        (r["split"], r["fraction"], r["record_index"]): r["p_goal"]
        for r in rows
        if abs(float(r["dose"]) - 1.0) < 1e-12
    }
    for row in rows:
        row["delta_vs_dose_1"] = float(row["p_goal"]) - float(
            one_lookup.get((row["split"], row["fraction"], row["record_index"]), row["p_goal"])
        )

    episode_summary = _episode_summaries(rows)
    curve_summary = _aggregate(rows, ["split", "fraction", "dose"])
    by_category = _aggregate(rows, ["split", "fraction", "category", "dose"])
    by_config = _aggregate(rows, ["split", "fraction", "config", "dose"])

    write_csv(out_dir / "dose_response_episode_level.csv", rows)
    write_csv(out_dir / "dose_response_episode_summary.csv", episode_summary)
    write_csv(out_dir / "dose_response_summary.csv", curve_summary)
    write_csv(out_dir / "dose_response_by_category.csv", by_category)
    write_csv(out_dir / "dose_response_by_config.csv", by_config)
    _plot(curve_summary, out_dir / "plots" / "dose_response_curve.png")
    write_text(out_dir / "summary.md", _summary_text(episode_summary, curve_summary))
    print(f"Belief dose-response diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
