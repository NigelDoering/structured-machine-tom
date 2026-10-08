"""
simulator-level counterfactual rollout diagnostics.

This script resamples goals and routes under modified mental states, then
evaluates the model on the regenerated trajectories. It is intentionally a
diagnostic approximation: the simulator does not expose a native "replay this
episode with edited latent state" API, so we reconstruct the BDI sampling step
from each record's supervised desire vector and belief vector.
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
    read_csv_dicts,
    resolve,
    safe_mean,
    write_csv,
    write_text,
)
from simulation.episode import EpisodeRecord
from simulation.router import Router


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True simulator counterfactual rollouts.")
    p.add_argument("--bnk-ckpt", default="outputs/diagnostics/sparse/bdi_bottleneck/seed_0/best.pt")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--splits", nargs="+", default=["test", "test_new_agent"])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/true_counterfactual_rollouts")
    p.add_argument("--max-records", type=int, default=0, help="Optional cap per split; 0 means all records.")
    p.add_argument("--record-start", type=int, default=0, help="Start offset within each selected split.")
    p.add_argument("--record-limit", type=int, default=0, help="Chunk size after --record-start; 0 means no chunk limit.")
    p.add_argument("--chunk-label", default="", help="Deterministic label written to chunk output rows.")
    p.add_argument("--router-k", type=int, default=5, help="Number of candidate router paths; 1 uses shortest-path only.")
    p.add_argument("--router-temperature", type=float, default=100.0, help="Router Boltzmann temperature.")
    p.add_argument("--merge-chunks", default="", help="Directory containing chunk CSVs to merge, skipping model evaluation.")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=32)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _slice_records(records: list[EpisodeRecord], args: argparse.Namespace) -> list[EpisodeRecord]:
    start = max(0, int(args.record_start))
    limit = 0
    if args.record_limit > 0 and args.max_records > 0:
        limit = min(args.record_limit, args.max_records)
    elif args.record_limit > 0:
        limit = args.record_limit
    elif args.max_records > 0:
        limit = args.max_records
    if limit > 0:
        return records[start : start + limit]
    return records[start:]


def _sample_goal(
    desire_vector: list[float],
    belief_vector: list[int],
    candidate_indices: list[int],
    meta: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[str, int] | None:
    weights = np.asarray([desire_vector[i] * belief_vector[i] for i in candidate_indices], dtype=np.float64)
    if weights.sum() <= 0:
        return None
    weights = weights / weights.sum()
    local_idx = int(rng.choice(np.arange(len(candidate_indices)), p=weights))
    poi_idx = candidate_indices[local_idx]
    poi_id = meta["idx_to_poi"][poi_idx]
    return poi_id, int(meta["poi_to_node"][poi_id])


def _route(router: Router, start_node: int, goal_node: int, rng: np.random.Generator) -> list[int] | None:
    path = router.route(start_node, goal_node, rng=rng)
    return path if path else None


def _make_cf_records(
    records: list[EpisodeRecord],
    meta: dict[str, Any],
    router: Router,
    cf_type: str,
    seed: int,
    record_start: int = 0,
    record_limit: int = 0,
    chunk_label: str = "",
) -> tuple[list[EpisodeRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    cf_records: list[EpisodeRecord] = []
    rows: list[dict[str, Any]] = []
    cf_meta: list[dict[str, Any]] = []
    categories = list(meta["category_to_indices"].keys())

    for idx, record in enumerate(records):
        source_index = record_start + idx
        belief = list(record.belief_vector)
        candidate_category = record.desire
        reason = ""
        row_base = {
            "record_index": source_index,
            "local_record_index": idx,
            "record_start": record_start,
            "record_limit": record_limit,
            "chunk_label": chunk_label,
            "cf_type": cf_type,
        }
        if cf_type == "belief":
            goal_idx = meta["poi_to_idx"][record.goal_poi]
            goal_cat = meta["idx_to_category"][goal_idx]
            candidate_category = goal_cat
            belief[goal_idx] = 0
            for alt_idx in meta["category_to_indices"][goal_cat]:
                if alt_idx != goal_idx:
                    belief[alt_idx] = 1
        elif cf_type == "desire":
            current = categories.index(record.desire) if record.desire in categories else 0
            alt_categories = categories[current + 1 :] + categories[:current]
            candidate_category = next(
                (
                    cat
                    for cat in alt_categories
                    if any(belief[i] == 1 for i in meta["category_to_indices"][cat])
                ),
                record.desire,
            )
        else:
            raise ValueError(f"Unknown cf_type: {cf_type}")

        sampled = _sample_goal(
            record.desire_vector,
            belief,
            meta["category_to_indices"][candidate_category],
            meta,
            rng,
        )
        if sampled is None:
            reason = "no_open_candidate"
            rows.append({**row_base, "success": 0, "failure_reason": reason})
            continue
        goal_poi, goal_node = sampled
        path = _route(router, record.start_node, goal_node, rng)
        if path is None:
            reason = "route_failed"
            rows.append({**row_base, "success": 0, "failure_reason": reason})
            continue

        data = record.to_dict()
        data["split"] = f"{record.split}_{cf_type}_cf"
        data["belief_vector"] = belief
        data["false_beliefs"] = [meta["idx_to_poi"][i] for i, value in enumerate(belief) if value == 0]
        data["desire"] = candidate_category
        data["goal_poi"] = goal_poi
        data["goal_node"] = goal_node
        data["path"] = path
        cf_records.append(EpisodeRecord.from_dict(data))
        cf_meta.append(row_base)
        rows.append(
            {
                **row_base,
                "success": 1,
                "failure_reason": "",
                "original_goal": record.goal_poi,
                "counterfactual_goal": goal_poi,
                "original_desire": record.desire,
                "counterfactual_desire": candidate_category,
            }
        )
    return cf_records, rows, cf_meta


def _evaluate_cf(
    model,
    cf_records: list[EpisodeRecord],
    split: str,
    cf_type: str,
    fractions: list[float],
    meta: dict[str, Any],
    node_coords: dict[int, tuple[float, float]],
    device: torch.device,
    args: argparse.Namespace,
    cf_meta: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    source_meta = cf_meta or [{} for _ in cf_records]
    for fraction in fractions:
        comps = collect_bdi_components(
            model,
            cf_records,
            node_coords,
            meta["poi_to_idx"],
            device,
            fraction=fraction,
            k=args.K,
            k_recent=args.K_recent,
            batch_size=args.batch,
            seed=args.seed,
        )
        logits = torch.stack([c["goal_logits"] for c in comps])
        probs = F.softmax(logits, dim=-1)
        labels = torch.tensor([int(c["goal_idx"]) for c in comps], dtype=torch.long)
        top5 = probs.topk(min(5, probs.size(1)), dim=1).indices
        for i, comp in enumerate(comps):
            source_row = source_meta[int(comp["record_index"])]
            rows.append(
                {
                    "split": split,
                    "cf_type": cf_type,
                    "fraction": fraction,
                    "record_index": source_row.get("record_index", comp["record_index"]),
                    "local_cf_record_index": comp["record_index"],
                    "local_record_index": source_row.get("local_record_index", comp["record_index"]),
                    "record_start": source_row.get("record_start", args.record_start),
                    "record_limit": source_row.get("record_limit", args.record_limit),
                    "chunk_label": source_row.get("chunk_label", args.chunk_label),
                    "agent_id": comp["agent_id"],
                    "config": comp["config"],
                    "counterfactual_goal": comp["goal_poi"],
                    "top1_correct": int(top5[i, 0].item() == labels[i].item()),
                    "top5_correct": int((top5[i] == labels[i]).any().item()),
                    "prob_counterfactual_goal": float(probs[i, labels[i]].item()),
                }
            )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["cf_type"], float(row["fraction"]))].append(row)
    out = []
    for (split, cf_type, fraction), vals in sorted(grouped.items()):
        out.append(
            {
                "split": split,
                "cf_type": cf_type,
                "fraction": fraction,
                "n": len(vals),
                "top1": safe_mean([float(v["top1_correct"]) for v in vals]),
                "top5": safe_mean([float(v["top5_correct"]) for v in vals]),
                "prob_counterfactual_goal": safe_mean([float(v["prob_counterfactual_goal"]) for v in vals]),
            }
        )
    return out


def _summary(success_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> str:
    n_total = len(success_rows)
    n_success = sum(1 for row in success_rows if int(row["success"]) == 1)
    chunks = sorted({row.get("chunk_label", "") for row in success_rows if row.get("chunk_label", "")})
    return "\n".join(
        [
            "# True Counterfactual Rollouts Summary",
            "",
            f"- Counterfactual rollout attempts: {n_total}.",
            f"- Successful routed rollouts: {n_success}.",
            f"- Evaluation rows: {len(metric_rows)}.",
            f"- Chunks merged/evaluated: {len(chunks)}.",
            "",
            "This diagnostic reconstructs simulator counterfactuals from recorded desire/belief labels. It is not a native replay API.",
        ]
    ) + "\n"


def _collect_chunk_rows(chunks_dir: str, out_dir, filename: str) -> list[dict[str, str]]:
    chunk_root = resolve(chunks_dir)
    rows: list[dict[str, str]] = []
    for path in sorted(chunk_root.rglob(filename)):
        if path.parent == out_dir or out_dir in path.parents:
            continue
        rows.extend(read_csv_dicts(path))
    return rows


def _merge_chunks(chunks_dir: str, out_dir) -> None:
    success_rows = _collect_chunk_rows(chunks_dir, out_dir, "cf_rollout_attempts.csv")
    belief_rows = _collect_chunk_rows(chunks_dir, out_dir, "cf_rollout_belief.csv")
    desire_rows = _collect_chunk_rows(chunks_dir, out_dir, "cf_rollout_desire.csv")
    metric_rows = belief_rows + desire_rows
    write_csv(out_dir / "cf_rollout_goal_accuracy.csv", _aggregate(metric_rows))
    write_csv(out_dir / "cf_rollout_belief.csv", belief_rows)
    write_csv(out_dir / "cf_rollout_desire.csv", desire_rows)
    write_csv(out_dir / "cf_rollout_attempts.csv", success_rows)
    write_text(out_dir / "summary.md", _summary(success_rows, metric_rows))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.fractions = args.fractions[:1]
        if args.max_records <= 0 and args.record_limit <= 0:
            args.record_limit = args.smoke_n
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_chunks:
        _merge_chunks(args.merge_chunks, out_dir)
        print(f"Merged true counterfactual rollout chunks into {out_dir}")
        return

    device = get_device(args.no_cuda)
    meta = poi_metadata()
    records_by_split = load_split_records(args.splits_dir, args.splits, smoke=False)
    if not records_by_split:
        raise SystemExit(f"No requested split files found under {resolve(args.splits_dir)}")
    ckpt_path = resolve(args.bnk_ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    model, _ = load_bdi_bottleneck(ckpt_path, device)
    node_coords = node_coords_from_graph()
    router = Router(
        graph_path=resolve("data/ucsd_walk.graphml"),
        k=args.router_k,
        temperature=args.router_temperature,
    )

    success_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        records = _slice_records(records, args)
        for cf_type in ["belief", "desire"]:
            cf_records, rows, cf_meta = _make_cf_records(
                records,
                meta,
                router,
                cf_type,
                args.seed,
                record_start=args.record_start,
                record_limit=args.record_limit,
                chunk_label=args.chunk_label,
            )
            for row in rows:
                row["split"] = split
            success_rows.extend(rows)
            if cf_records:
                metric_rows.extend(
                    _evaluate_cf(
                        model,
                        cf_records,
                        split,
                        cf_type,
                        args.fractions,
                        meta,
                        node_coords,
                        device,
                        args,
                        cf_meta,
                    )
                )

    belief_rows = [r for r in metric_rows if r["cf_type"] == "belief"]
    desire_rows = [r for r in metric_rows if r["cf_type"] == "desire"]
    write_csv(out_dir / "cf_rollout_goal_accuracy.csv", _aggregate(metric_rows))
    write_csv(out_dir / "cf_rollout_belief.csv", belief_rows)
    write_csv(out_dir / "cf_rollout_desire.csv", desire_rows)
    write_csv(out_dir / "cf_rollout_attempts.csv", success_rows)
    write_text(out_dir / "summary.md", _summary(success_rows, metric_rows))
    print(f"True counterfactual rollout diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
