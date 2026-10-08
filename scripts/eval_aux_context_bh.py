"""
Regenerate the paper goal-inference table with the AuxContext baseline.

The main goal-inference script predates the auxiliary-supervised context
baseline. This helper keeps the existing paper results intact, evaluates
AuxContext checkpoints on the same quarter fractions, and rewrites the
belief-heldout Table 1 used by the manuscript.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.context_goal_predictor import AuxSupervisedContextGoalPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from scripts.diag_utils import write_csv, write_text

import scripts.train_bdi_predictor as _tbp


REPORT_FRACS = [0.25, 0.50, 0.75, 1.00]
EVAL_SPLITS = ["test", "test_new_agent", "test_bh", "test_new_agent_bh"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add AuxContext to paper goal-inference table.")
    p.add_argument("--aux-root", default="outputs/context_goal_aux")
    p.add_argument("--base-results", default="results/goal_inference/results.json")
    p.add_argument("--splits", default="data/splits")
    p.add_argument("--out-table", default="results/goal_inference/top1_bh_table.tex")
    p.add_argument("--out-csv", default="results/goal_inference/aux_context_goal_quarters.csv")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 5, 10])
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _device(no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_aux_checkpoint(path: Path, device: torch.device) -> AuxSupervisedContextGoalPredictor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_kwargs = ckpt.get("model_kwargs") or {
        "num_pois": 24,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 3,
        "dim_ff": 256,
        "dropout": 0.1,
    }
    model = AuxSupervisedContextGoalPredictor(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_records(path: Path) -> list[Any]:
    from simulation.episode import EpisodeRecord

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_json(line))
    return records


def _evaluate_aux(args: argparse.Namespace) -> dict[str, dict[float, dict[str, float]]]:
    device = _device(args.no_cuda)
    splits_dir = _resolve(args.splits)
    aux_root = _resolve(args.aux_root)

    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    build_poi_index(poi_cfg)
    poi_order = [poi["id"] for cat_data in poi_cfg["categories"].values() for poi in cat_data["pois"]]
    poi_to_idx = {pid: i for i, pid in enumerate(poi_order)}

    import osmnx as ox

    graph = ox.load_graphml(ROOT / "data" / "ucsd_walk.graphml")
    node_coords = node_coords_from_graph(graph)

    records_by_split = {
        split: _load_records(splits_dir / f"{split}.jsonl")
        for split in EVAL_SPLITS
        if (splits_dir / f"{split}.jsonl").exists()
    }

    old_fractions = list(_tbp.FRACTIONS)
    _tbp.FRACTIONS = list(REPORT_FRACS)
    try:
        per_seed = []
        for seed in args.seeds:
            ckpt_path = aux_root / f"seed_{seed}" / "best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing AuxContext checkpoint: {ckpt_path}")
            model = _load_aux_checkpoint(ckpt_path, device)
            seed_result: dict[str, dict[float, dict[str, float]]] = {}
            for split, records in records_by_split.items():
                seed_result[split] = _tbp.eval_at_fractions(
                    model,
                    records,
                    node_coords,
                    poi_to_idx,
                    device,
                    K=args.K,
                    K_recent=args.K_recent,
                    batch_size=args.batch,
                    rng_seed=seed,
                )
            per_seed.append(seed_result)
    finally:
        _tbp.FRACTIONS = old_fractions

    agg: dict[str, dict[float, dict[str, float]]] = {}
    for split in records_by_split:
        agg[split] = {}
        for frac in REPORT_FRACS:
            top1 = [seed_result[split][frac]["top1"] for seed_result in per_seed]
            top5 = [seed_result[split][frac]["top5"] for seed_result in per_seed]
            agg[split][frac] = {
                "top1": float(np.mean(top1)),
                "top5": float(np.mean(top5)),
                "top1_std": float(np.std(top1, ddof=0)),
                "top5_std": float(np.std(top5, ddof=0)),
            }
    return agg


def _base_value(results: dict[str, Any], model: str, split: str, frac: float, metric: str) -> float:
    split_data = results[model][split]
    return float((split_data.get(frac) or split_data.get(str(frac)))[metric])


def _write_aux_csv(aux: dict[str, dict[float, dict[str, float]]], out_path: Path) -> None:
    rows = []
    for split, frac_data in aux.items():
        for frac, metrics in frac_data.items():
            rows.append(
                {
                    "model": "AuxSupervisedContextGoalPredictor",
                    "split": split,
                    "fraction": frac,
                    **metrics,
                }
            )
    write_csv(out_path, rows)


def _write_bh_table(base: dict[str, Any], aux: dict[str, dict[float, dict[str, float]]], out_path: Path) -> None:
    labels = {
        "VisitFrequency": "VisitFrequency",
        "BToMInformed": r"\textit{BToM (Informed) [approximate upper bound]}",
        "GoalPredictor": "GoalPredictor",
        "ContextGoalPredictor": "ContextGoalPredictor",
        "AuxSupervisedContextGoalPredictor": r"\shortstack[l]{AuxContext\\(aux. sup.)}",
        "BDIBottleneck": "BDIBottleneck (Ours)",
    }
    order = [
        "VisitFrequency",
        "BToMInformed",
        "GoalPredictor",
        "ContextGoalPredictor",
        "AuxSupervisedContextGoalPredictor",
        "BDIBottleneck",
    ]
    splits = ["test_bh", "test_new_agent_bh"]

    rows: list[tuple[str, str, list[float]]] = []
    for model in order:
        values = []
        for split in splits:
            for frac in REPORT_FRACS:
                if model == "AuxSupervisedContextGoalPredictor":
                    values.append(aux[split][frac]["top1"] * 100.0)
                else:
                    values.append(_base_value(base, model, split, frac, "top1") * 100.0)
        rows.append((model, labels[model], values))

    non_ceiling = [vals for model, _, vals in rows if model != "BToMInformed"]
    col_max = [max(vals[i] for vals in non_ceiling) for i in range(len(REPORT_FRACS) * 2)]

    def fmt(model: str, value: float, col: int) -> str:
        text = f"{value:.2f}"
        if model == "BToMInformed":
            return r"\underline{" + text + "}"
        if abs(value - col_max[col]) < 1e-8:
            return r"\textbf{" + text + "}"
        return text

    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.5pt}",
        (
            r"  \caption{Top-1 accuracy (\%) at 25/50/75/100\% path reveal for belief heldout splits."
            r" AuxContext is the unbottlenecked context model trained with the same auxiliary desire/belief losses as BDIBottleneck."
            r" \underline{Underline}: BToM (Informed), approximate upper bound on achievable accuracy."
            r" \textbf{Bold}: best result per column among learned models.}"
        ),
        r"  \label{tab:goal_inference_top1_quarters_bh}",
        r"  \begin{tabular}{lrrrrrrrr}",
        r"    \toprule",
        (
            r"    & \multicolumn{4}{c}{Test (belief heldout)}"
            r" & \multicolumn{4}{c}{Test (new agent, belief heldout)} \\"
        ),
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"    Model & 25\% & 50\% & 75\% & 100\% & 25\% & 50\% & 75\% & 100\% \\",
        r"    \midrule",
    ]
    for model, label, values in rows:
        cells = " & ".join(fmt(model, value, i) for i, value in enumerate(values))
        lines.append(f"    {label} & {cells} \\\\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    write_text(out_path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    base_path = _resolve(args.base_results)
    with open(base_path) as f:
        base = json.load(f)
    aux = _evaluate_aux(args)
    _write_aux_csv(aux, _resolve(args.out_csv))
    _write_bh_table(base, aux, _resolve(args.out_table))


if __name__ == "__main__":
    main()
