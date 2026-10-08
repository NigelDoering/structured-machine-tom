"""
dense false-belief configuration summaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from diag_utils import (
    load_json,
    load_split_records,
    poi_metadata,
    resolve,
    safe_mean,
    write_csv,
    write_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense false-belief summaries.")
    p.add_argument("--belief-config", default="config/belief_configs_dense.json")
    p.add_argument("--splits-dir", default="outputs/diagnostics/dense/splits")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test", "test_new_agent", "test_bh", "test_new_agent_bh"])
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/dense_false_beliefs")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=48)
    return p.parse_args()


def _density(count: int) -> str:
    if count <= 5:
        return "sparse"
    if count <= 8:
        return "medium"
    return "dense"


def _config_rows(cfg: dict[str, Any], meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section in ["training", "held_out"]:
        for name, item in cfg.get(section, {}).items():
            false = list(item.get("false_beliefs", []))
            by_cat = defaultdict(int)
            for poi_id in false:
                by_cat[meta["poi_to_category"][poi_id]] += 1
            rows.append(
                {
                    "section": section,
                    "config": name,
                    "n_closed": len(false),
                    "density": _density(len(false)),
                    **{f"closed_{cat}": by_cat.get(cat, 0) for cat in meta["category_to_indices"]},
                    "all_categories_have_open_poi": int(all(by_cat.get(cat, 0) < 4 for cat in meta["category_to_indices"])),
                    "false_beliefs": " ".join(false),
                }
            )
    return rows


def _split_density_rows(records_by_split) -> list[dict[str, Any]]:
    rows = []
    grouped = defaultdict(list)
    for split, records in records_by_split.items():
        for r in records:
            grouped[(split, r.belief_config)].append(sum(1 for v in r.belief_vector if v == 0))
    for (split, config), counts in sorted(grouped.items()):
        mean_count = safe_mean([float(c) for c in counts])
        rows.append(
            {
                "split": split,
                "config": config,
                "n_episodes": len(counts),
                "mean_closed_count": mean_count,
                "density": _density(int(round(mean_count))),
            }
        )
    return rows


def _summary(config_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> str:
    dense = [r for r in config_rows if r["density"] == "dense"]
    medium = [r for r in config_rows if r["density"] == "medium"]
    sparse = [r for r in config_rows if r["density"] == "sparse"]
    return "\n".join(
        [
            "# Dense False-Belief Summary",
            "",
            f"- Sparse configs: {len(sparse)}.",
            f"- Medium configs: {len(medium)}.",
            f"- Dense configs: {len(dense)}.",
            f"- Split/config density rows: {len(split_rows)}.",
            "",
            "Default sparse configs are unchanged; dense configs are used only when explicitly passed to simulation/training commands.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = poi_metadata()
    cfg = load_json(args.belief_config)
    config_rows = _config_rows(cfg, meta)
    records_by_split = load_split_records(args.splits_dir, args.splits, smoke=args.smoke, smoke_n=args.smoke_n)
    split_rows = _split_density_rows(records_by_split)

    write_csv(out_dir / "dense_config_summary.csv", config_rows)
    write_csv(out_dir / "attribution_by_density.csv", split_rows)
    write_csv(out_dir / "goal_accuracy_by_density.csv", [])
    write_csv(out_dir / "intervention_by_density.csv", [])
    write_csv(out_dir / "usage_by_density.csv", [])
    write_text(out_dir / "summary.md", _summary(config_rows, split_rows))
    print(f"Dense false-belief summaries written to {out_dir}")


if __name__ == "__main__":
    main()
