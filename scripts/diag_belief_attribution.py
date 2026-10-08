"""
belief attribution metrics and label imbalance.

The belief vector convention is 1=open, 0=believed closed. Closed-class metrics
therefore use p_closed = 1 - sigmoid(belief_logits).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from diag_utils import (
    ROOT,
    binary_metrics_for_closed,
    calibration_bins,
    collect_bdi_components,
    get_device,
    load_bdi_bottleneck,
    load_split_records,
    node_coords_from_graph,
    poi_metadata,
    resolve,
    write_csv,
    write_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Belief attribution diagnostics.")
    p.add_argument("--bnk-ckpt", default="outputs/diagnostics/sparse/bdi_bottleneck/seed_0/best.pt")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--splits", nargs="+", default=["val", "test", "test_new_agent", "test_bh", "test_new_agent_bh"])
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/current_belief_attribution")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=48)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _train_evidence_counts(splits_dir: str | Path) -> dict[tuple[str, str], int]:
    from diag_utils import load_records

    train_path = resolve(splits_dir) / "train.jsonl"
    counts: dict[tuple[str, str], int] = defaultdict(int)
    if not train_path.exists():
        return counts
    for record in load_records(train_path):
        counts[(record.agent_id, record.goal_poi)] += 1
    return counts


def _metric_row(group: str, value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([r["target_open"] for r in rows], dtype=np.int32)
    p = np.asarray([r["pred_open_prob"] for r in rows], dtype=np.float64)
    metrics = binary_metrics_for_closed(y, p)
    return {"group": group, "value": value, **metrics}


def _group_metrics(entry_rows: list[dict[str, Any]], key: str, label: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entry_rows:
        grouped[str(row[key])].append(row)
    return [_metric_row(label, value, rows) for value, rows in sorted(grouped.items())]


def _imbalance_rows(entry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [
        ("all", lambda r: "all"),
        ("split", lambda r: r["split"]),
        ("config", lambda r: r["config"]),
        ("category", lambda r: r["category"]),
        ("closed_count", lambda r: str(r["closed_count"])),
    ]
    out: list[dict[str, Any]] = []
    for group_name, fn in groups:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in entry_rows:
            grouped[str(fn(row))].append(row)
        for value, rows in sorted(grouped.items()):
            n_open = sum(1 for r in rows if r["target_open"] == 1)
            n_closed = sum(1 for r in rows if r["target_open"] == 0)
            total = n_open + n_closed
            out.append(
                {
                    "group": group_name,
                    "value": value,
                    "n_entries": total,
                    "n_open": n_open,
                    "n_closed": n_closed,
                    "closed_prop": n_closed / total if total else 0.0,
                }
            )
    return out


def _summary_text(overall_rows: list[dict[str, Any]], imbalance: list[dict[str, Any]]) -> str:
    all_row = next((r for r in overall_rows if r["group"] == "all"), None)
    val_rows = [r for r in overall_rows if r["group"] == "split"]
    lines = [
        "# Current Belief Attribution Summary",
        "",
        "Closed-class metrics treat false beliefs as the positive class.",
        "",
    ]
    if all_row:
        lines += [
            f"- Overall belief accuracy: {all_row['overall_accuracy']:.4f}.",
            f"- Closed-class precision/recall/F1: {all_row['closed_precision']:.4f} / {all_row['closed_recall']:.4f} / {all_row['closed_f1']:.4f}.",
            f"- Macro-F1: {all_row['macro_f1']:.4f}; balanced accuracy: {all_row['balanced_accuracy']:.4f}.",
            f"- Closed-entry proportion: {all_row['closed_prop']:.4f} ({all_row['n_closed']} closed of {all_row['n_entries']} entries).",
        ]
    if val_rows:
        lines.append("")
        lines.append("## By Split")
        for row in val_rows:
            lines.append(
                f"- {row['value']}: closed F1={row['closed_f1']:.4f}, "
                f"PR-AUC={row['pr_auc_closed']:.4f}, closed_prop={row['closed_prop']:.4f}."
            )
    all_imb = next((r for r in imbalance if r["group"] == "all"), None)
    if all_imb and all_imb["closed_prop"] < 0.25:
        lines += [
            "",
            "## Interpretation",
            "- The label distribution is sparse and majority-open, so overall belief accuracy is expected to overstate closed-class performance.",
            "- Low closed-class F1 should be interpreted together with PR-AUC, closed-entry prevalence, and evidence-conditioned breakdowns.",
            "- This file is a diagnostic summary.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.no_cuda)
    meta = poi_metadata()
    poi_to_idx = meta["poi_to_idx"]
    idx_to_poi = meta["idx_to_poi"]
    idx_to_category = meta["idx_to_category"]

    records_by_split = load_split_records(args.splits_dir, args.splits, smoke=args.smoke, smoke_n=args.smoke_n)
    if not records_by_split:
        raise SystemExit(f"No requested split files found under {resolve(args.splits_dir)}")

    ckpt_path = resolve(args.bnk_ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    model, _ = load_bdi_bottleneck(ckpt_path, device)
    node_coords = node_coords_from_graph()
    evidence_counts = _train_evidence_counts(args.splits_dir)

    entry_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for split_name, records in records_by_split.items():
        components = collect_bdi_components(
            model,
            records,
            node_coords,
            poi_to_idx,
            device,
            fraction=1.0,
            k=args.K,
            k_recent=args.K_recent,
            batch_size=args.batch,
            seed=args.seed,
        )
        for comp in components:
            target = comp["belief_target"].numpy()
            pred = comp["belief_probs"].numpy()
            episode_rows.append(
                {
                    "split": split_name,
                    "record_index": comp["record_index"],
                    "agent_id": comp["agent_id"],
                    "character": comp["character"],
                    "config": comp["config"],
                    "goal_poi": comp["goal_poi"],
                    "closed_count": comp["closed_count"],
                    "mean_pred_closed": float((1.0 - pred).mean()),
                    "true_closed_count": int((target == 0).sum()),
                }
            )
            for poi_idx, poi_id in idx_to_poi.items():
                category = idx_to_category[poi_idx]
                entry_rows.append(
                    {
                        "split": split_name,
                        "record_index": comp["record_index"],
                        "agent_id": comp["agent_id"],
                        "character": comp["character"],
                        "config": comp["config"],
                        "goal_poi": comp["goal_poi"],
                        "poi_id": poi_id,
                        "poi_idx": poi_idx,
                        "category": category,
                        "closed_count": comp["closed_count"],
                        "target_open": int(target[poi_idx]),
                        "target_closed": int(target[poi_idx] == 0),
                        "pred_open_prob": float(pred[poi_idx]),
                        "pred_closed_prob": float(1.0 - pred[poi_idx]),
                        "train_goal_evidence": evidence_counts.get((comp["agent_id"], poi_id), 0),
                    }
                )

    overall_rows = [_metric_row("all", "all", entry_rows)]
    overall_rows += _group_metrics(entry_rows, "split", "split")
    by_config = _group_metrics(entry_rows, "config", "config")
    by_poi = _group_metrics(entry_rows, "poi_id", "poi")
    by_category = _group_metrics(entry_rows, "category", "category")
    by_closed_count = _group_metrics(entry_rows, "closed_count", "closed_count")
    by_evidence = _group_metrics(
        [
            {**row, "evidence_bin": "0" if row["train_goal_evidence"] == 0 else "1-3" if row["train_goal_evidence"] <= 3 else "4-8" if row["train_goal_evidence"] <= 8 else "9+"}
            for row in entry_rows
        ],
        "evidence_bin",
        "evidence_bin",
    )
    imbalance = _imbalance_rows(entry_rows)

    cal_rows: list[dict[str, Any]] = []
    for row in overall_rows:
        if row["group"] == "split":
            rows = [r for r in entry_rows if r["split"] == row["value"]]
        elif row["group"] == "all":
            rows = entry_rows
        else:
            continue
        y = np.asarray([r["target_open"] for r in rows], dtype=np.int32)
        p = np.asarray([r["pred_open_prob"] for r in rows], dtype=np.float64)
        for cal in calibration_bins(y, p):
            cal_rows.append({"group": row["group"], "value": row["value"], **cal})

    write_csv(out_dir / "metrics_overall.csv", overall_rows)
    write_csv(out_dir / "metrics_by_config.csv", by_config)
    write_csv(out_dir / "metrics_by_poi.csv", by_poi)
    write_csv(out_dir / "metrics_by_category.csv", by_category)
    write_csv(out_dir / "metrics_by_closed_count.csv", by_closed_count)
    write_csv(out_dir / "metrics_by_evidence.csv", by_evidence)
    write_csv(out_dir / "label_imbalance.csv", imbalance)
    write_csv(out_dir / "calibration.csv", cal_rows)
    write_csv(out_dir / "episode_level.csv", episode_rows)
    write_csv(out_dir / "entry_level.csv", entry_rows)
    write_text(out_dir / "summary.md", _summary_text(overall_rows, imbalance))

    print(f"Belief attribution diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
