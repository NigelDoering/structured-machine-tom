"""
Refresh paper-facing goal-inference figures with the AuxContext baseline.

This script does not retrain or re-evaluate the original paper models. It
merges the existing goal-inference results with the completed AuxContext
paper runs, then regenerates the figures and LaTeX tables used by the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_goal_inference as goal_eval


AUX_KEY = "AuxSupervisedContextGoalPredictor"
MODEL_ORDER = [
    "VisitFrequency",
    "BToMInformed",
    "GoalPredictor",
    "ContextGoalPredictor",
    AUX_KEY,
    "BDIBottleneck",
]
TENTH_FRACS = [round(i / 10.0, 1) for i in range(1, 11)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-results", default="results/goal_inference/results.json")
    p.add_argument("--aux-root", default="outputs/context_goal_aux")
    p.add_argument("--aux-quarter-csv", default="results/goal_inference/aux_context_goal_quarters.csv")
    p.add_argument("--out-dir", default="results/goal_inference")
    p.add_argument("--paper", action="store_true", help="Write 300 DPI paper figures.")
    return p.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_results(path: Path) -> dict[str, dict[str, dict[float, dict[str, float]]]]:
    with open(path) as f:
        raw = json.load(f)
    results: dict[str, dict[str, dict[float, dict[str, float]]]] = {}
    for model, split_data in raw.items():
        results[model] = {}
        for split, frac_data in split_data.items():
            results[model][split] = {float(frac): metrics for frac, metrics in frac_data.items()}
    return results


def _json_ready(results: dict[str, dict[str, dict[float, dict[str, float]]]]) -> dict[str, Any]:
    return {
        model: {
            split: {str(frac): metrics for frac, metrics in frac_data.items()}
            for split, frac_data in split_data.items()
        }
        for model, split_data in results.items()
    }


def _load_aux_seed_results(aux_root: Path) -> list[dict[str, dict[float, dict[str, float]]]]:
    per_seed = []
    for seed_dir in sorted(aux_root.glob("seed_*")):
        result_path = seed_dir / "accuracy_results.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            raw = json.load(f)["goal_fractions"]
        per_seed.append(
            {
                split: {float(frac): metrics for frac, metrics in frac_data.items()}
                for split, frac_data in raw.items()
            }
        )
    if not per_seed:
        raise FileNotFoundError(f"No AuxContext accuracy_results.json files found under {aux_root}")
    return per_seed


def _aggregate_aux(per_seed: list[dict[str, dict[float, dict[str, float]]]]) -> dict[str, dict[float, dict[str, float]]]:
    split_names = sorted(set().union(*(seed.keys() for seed in per_seed)))
    aggregate: dict[str, dict[float, dict[str, float]]] = {}
    for split in split_names:
        fracs = sorted(set().union(*(seed.get(split, {}).keys() for seed in per_seed)))
        aggregate[split] = {}
        for frac in fracs:
            top1 = [seed[split][frac]["top1"] for seed in per_seed if split in seed and frac in seed[split]]
            top5 = [seed[split][frac]["top5"] for seed in per_seed if split in seed and frac in seed[split]]
            aggregate[split][frac] = {
                "top1": float(np.mean(top1)),
                "top5": float(np.mean(top5)),
                "top1_std": float(np.std(top1, ddof=0)),
                "top5_std": float(np.std(top5, ddof=0)),
            }
    return aggregate


def _merge_aux_quarters(aux_results: dict[str, dict[float, dict[str, float]]], csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing AuxContext quarter-fraction CSV: {csv_path}")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["model"] != AUX_KEY:
                continue
            split = row["split"]
            frac = float(row["fraction"])
            aux_results.setdefault(split, {})[frac] = {
                "top1": float(row["top1"]),
                "top5": float(row["top5"]),
                "top1_std": float(row["top1_std"]),
                "top5_std": float(row["top5_std"]),
            }


def _ordered_results(
    results: dict[str, dict[str, dict[float, dict[str, float]]]]
) -> dict[str, dict[str, dict[float, dict[str, float]]]]:
    ordered = {model: results[model] for model in MODEL_ORDER if model in results}
    ordered.update({model: data for model, data in results.items() if model not in ordered})
    return ordered


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}"


def _write_tenths_table(
    results: dict[str, dict[str, dict[float, dict[str, float]]]],
    out_path: Path,
) -> None:
    models = ["GoalPredictor", "ContextGoalPredictor", AUX_KEY, "BDIBottleneck"]
    labels = {
        "GoalPredictor": "Goal",
        "ContextGoalPredictor": "Ctx",
        AUX_KEY: "Aux",
        "BDIBottleneck": "BDI",
    }
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.2pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \caption{Top-1 goal inference accuracy (\%) at 10\% path-fraction increments. Aux is the matched auxiliary-supervised context baseline.}",
        r"  \label{tab:goal_inference_top1_tenths}",
        r"  \begin{tabular}{r*{8}{r}}",
        r"    \toprule",
        r"    Path & \multicolumn{4}{c}{Test} & \multicolumn{4}{c}{Test (new agent)} \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        "    frac. (\\%) & "
        + " & ".join(labels[m] for m in models)
        + " & "
        + " & ".join(labels[m] for m in models)
        + r" \\",
        r"    \midrule",
    ]
    for frac in TENTH_FRACS:
        cells = [f"{int(frac * 100):>3}"]
        for split in ("test", "test_new_agent"):
            for model in models:
                value = results[model][split][frac]["top1"]
                cells.append(_fmt_pct(value))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))
    print(f"  Saved -> {out_path}")


def _assert_aux_complete(results: dict[str, dict[str, dict[float, dict[str, float]]]]) -> None:
    required_splits = {"test", "test_new_agent", "test_bh", "test_new_agent_bh"}
    missing_splits = required_splits - set(results[AUX_KEY])
    if missing_splits:
        raise ValueError(f"AuxContext missing splits: {sorted(missing_splits)}")
    for split in required_splits:
        missing_fracs = set(goal_eval.PLOT_FRACTIONS) - set(results[AUX_KEY][split])
        if missing_fracs:
            raise ValueError(f"AuxContext {split} missing plot fractions: {sorted(missing_fracs)}")


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(_resolve(args.base_results))
    aux_results = _aggregate_aux(_load_aux_seed_results(_resolve(args.aux_root)))
    _merge_aux_quarters(aux_results, _resolve(args.aux_quarter_csv))
    results[AUX_KEY] = aux_results
    results = _ordered_results(results)
    _assert_aux_complete(results)

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(_json_ready(results), indent=2))
    print(f"Results -> {results_path}")

    goal_eval.plot_top1_comparison_and_gap(
        results,
        "test",
        "test_new_agent",
        out_dir / "top1_comparison.png",
        out_dir / "top1_comparison_gap.png",
        paper=args.paper,
    )
    goal_eval.plot_top1_combined(
        results,
        "test",
        "test_new_agent",
        out_dir / "top1_combined.png",
        paper=args.paper,
    )
    goal_eval.plot_top1_comparison_and_gap(
        results,
        "test_bh",
        "test_new_agent_bh",
        out_dir / "top1_comparison_bh.png",
        out_dir / "top1_comparison_gap_bh.png",
        title_suffix=" (BH)",
        paper=args.paper,
    )
    goal_eval.plot_top1_combined(
        results,
        "test_bh",
        "test_new_agent_bh",
        out_dir / "top1_combined_bh.png",
        title_suffix=" (BH)",
        paper=args.paper,
    )
    goal_eval.plot_top5_split(results, "test", out_dir / "top5_test.png", paper=args.paper)
    goal_eval.plot_top5_split(results, "test_new_agent", out_dir / "top5_test_new_agent.png", paper=args.paper)
    goal_eval._generate_bh_table_tex(results, out_dir / "top1_bh_table.tex")
    _write_tenths_table(results, out_dir / "top1_tenths_table.tex")


if __name__ == "__main__":
    main()
