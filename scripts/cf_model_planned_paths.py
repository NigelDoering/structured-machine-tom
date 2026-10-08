"""
model-planned full counterfactual path diagnostics.

This diagnostic freezes BDIBottleneck, edits the belief/desire channels at an
observed trajectory prefix, lets the model choose a counterfactual goal, and
routes a new future suffix from the intervention node to that model-chosen
goal. It complements cf_simulator_rollouts.py, which samples
a simulator counterfactual goal first and evaluates whether the model predicts
that generated goal.
"""
from __future__ import annotations

import argparse
import json
import math
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


CHANNEL_MODES = ["predicted_channel", "oracle_channel"]
CF_TYPES = ["belief", "desire"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-planned counterfactual path diagnostics.")
    p.add_argument("--bnk-ckpt", default="outputs/diagnostics/sparse/bdi_bottleneck/seed_0/best.pt")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--splits", nargs="+", default=["test", "test_new_agent"])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--out-dir", default="outputs/diagnostics/analyses/model_planned_counterfactual_paths")
    p.add_argument("--max-records", type=int, default=0, help="Optional cap per split after --record-start; 0 means all records.")
    p.add_argument("--record-start", type=int, default=0, help="Start offset within each selected split.")
    p.add_argument("--record-limit", type=int, default=0, help="Chunk size after --record-start; 0 means no chunk limit.")
    p.add_argument("--chunk-label", default="", help="Deterministic label written to chunk output rows.")
    p.add_argument("--router-k", type=int, default=1, help="Number of candidate router paths; 1 uses shortest-path only.")
    p.add_argument("--router-temperature", type=float, default=100.0)
    p.add_argument("--merge-chunks", default="", help="Directory containing chunk CSVs to merge, skipping model evaluation.")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--K-recent", type=int, default=5, dest="K_recent")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=16)
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
    return records[start : start + limit] if limit > 0 else records[start:]


def _prefix_len(path: list[int], fraction: float) -> int:
    return max(1, min(len(path), round(float(fraction) * len(path))))


def _edges(path: list[int]) -> set[tuple[int, int]]:
    return set(zip(path, path[1:]))


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    denom = len(left | right)
    return float(len(left & right) / denom) if denom else 0.0


def _path_length(router: Router, path: list[int]) -> float:
    if len(path) <= 1:
        return 0.0
    total = 0.0
    for u, v in zip(path, path[1:]):
        try:
            total += float(router.G[u][v].get("length", 1.0))
        except Exception:
            return float("nan")
    return total


def _join_path(path: list[int]) -> str:
    return " ".join(str(node) for node in path)


def _topk_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    top5 = probs.topk(min(5, probs.size(-1)), dim=-1).indices
    return probs, top5


def _next_feasible_category(record: EpisodeRecord, meta: dict[str, Any], belief: list[int]) -> str:
    categories = list(meta["category_to_indices"].keys())
    current = categories.index(record.desire) if record.desire in categories else 0
    for cat in categories[current + 1 :] + categories[:current]:
        if any(belief[i] == 1 for i in meta["category_to_indices"][cat]):
            return cat
    return record.desire


def _renorm_to_category(values: torch.Tensor, category: str, meta: dict[str, Any]) -> torch.Tensor:
    out = torch.zeros_like(values)
    indices = meta["category_to_indices"][category]
    mass = values[indices].sum()
    if float(mass.item()) > 1e-9:
        out[indices] = values[indices] / mass
    else:
        out[indices] = 1.0 / len(indices)
    return out


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


def _edited_channels(
    comp: dict[str, Any],
    record: EpisodeRecord,
    meta: dict[str, Any],
    cf_type: str,
    channel_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int], str]:
    if channel_mode == "predicted_channel":
        desire = comp["desire_probs"].clone().float()
        belief = comp["belief_probs"].clone().float()
    elif channel_mode == "oracle_channel":
        desire = comp["desire_target"].clone().float()
        belief = comp["belief_target"].clone().float()
    else:
        raise ValueError(f"Unknown channel_mode: {channel_mode}")

    sim_belief = list(record.belief_vector)
    candidate_category = record.desire
    goal_idx = meta["poi_to_idx"][record.goal_poi]

    if cf_type == "belief":
        goal_cat = meta["idx_to_category"][goal_idx]
        candidate_category = goal_cat
        belief[goal_idx] = 0.0
        sim_belief[goal_idx] = 0
        for alt_idx in meta["category_to_indices"][goal_cat]:
            if alt_idx != goal_idx:
                belief[alt_idx] = 1.0
                sim_belief[alt_idx] = 1
    elif cf_type == "desire":
        candidate_category = _next_feasible_category(record, meta, sim_belief)
        desire = _renorm_to_category(desire, candidate_category, meta)
    else:
        raise ValueError(f"Unknown cf_type: {cf_type}")

    return desire, belief, sim_belief, candidate_category


def _simulator_counterfactual(
    record: EpisodeRecord,
    meta: dict[str, Any],
    router: Router,
    cf_type: str,
    intervention_node: int,
    source_index: int,
    fraction: float,
    seed: int,
) -> dict[str, Any]:
    goal_idx = meta["poi_to_idx"][record.goal_poi]
    sim_belief = list(record.belief_vector)
    if cf_type == "belief":
        candidate_category = meta["idx_to_category"][goal_idx]
        sim_belief[goal_idx] = 0
        for alt_idx in meta["category_to_indices"][candidate_category]:
            if alt_idx != goal_idx:
                sim_belief[alt_idx] = 1
    elif cf_type == "desire":
        candidate_category = _next_feasible_category(record, meta, sim_belief)
    else:
        raise ValueError(f"Unknown cf_type: {cf_type}")

    frac_key = int(round(float(fraction) * 1000))
    cf_offset = 17 if cf_type == "belief" else 29
    rng = np.random.default_rng(seed + source_index * 1009 + frac_key * 37 + cf_offset)
    sampled = _sample_goal(
        record.desire_vector,
        sim_belief,
        meta["category_to_indices"][candidate_category],
        meta,
        rng,
    )
    if sampled is None:
        return {
            "simulator_success": 0,
            "simulator_failure_reason": "no_open_candidate",
            "simulator_cf_goal": "",
            "simulator_cf_goal_idx": "",
            "simulator_cf_path": [],
        }
    goal_poi, goal_node = sampled
    path = router.route(intervention_node, goal_node, rng=rng)
    if not path:
        return {
            "simulator_success": 0,
            "simulator_failure_reason": "route_failed",
            "simulator_cf_goal": goal_poi,
            "simulator_cf_goal_idx": meta["poi_to_idx"][goal_poi],
            "simulator_cf_path": [],
        }
    return {
        "simulator_success": 1,
        "simulator_failure_reason": "",
        "simulator_cf_goal": goal_poi,
        "simulator_cf_goal_idx": meta["poi_to_idx"][goal_poi],
        "simulator_cf_path": path,
    }


def _plan_path(router: Router, origin: int, goal_node: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return router.route(origin, goal_node, rng=rng)


def _path_metrics(router: Router, original_suffix: list[int], planned_suffix: list[int]) -> dict[str, float]:
    original_nodes = set(original_suffix)
    planned_nodes = set(planned_suffix)
    original_edges = _edges(original_suffix)
    planned_edges = _edges(planned_suffix)
    original_len = _path_length(router, original_suffix)
    planned_len = _path_length(router, planned_suffix)
    if math.isnan(original_len) or original_len <= 1e-9 or math.isnan(planned_len):
        ratio = float("nan")
    else:
        ratio = planned_len / original_len
    return {
        "original_suffix_nodes": len(original_suffix),
        "planned_suffix_nodes": len(planned_suffix),
        "original_suffix_length_m": original_len,
        "planned_suffix_length_m": planned_len,
        "route_length_ratio_original": ratio,
        "node_overlap_original": _jaccard(original_nodes, planned_nodes),
        "edge_overlap_original": _jaccard(original_edges, planned_edges),
    }


def _sim_path_metrics(router: Router, planned_suffix: list[int], sim_suffix: list[int]) -> dict[str, float]:
    if not sim_suffix:
        return {
            "node_overlap_simulator": float("nan"),
            "edge_overlap_simulator": float("nan"),
            "route_length_ratio_simulator": float("nan"),
        }
    planned_len = _path_length(router, planned_suffix)
    sim_len = _path_length(router, sim_suffix)
    ratio = planned_len / sim_len if sim_len > 1e-9 and not math.isnan(planned_len) else float("nan")
    return {
        "node_overlap_simulator": _jaccard(set(planned_suffix), set(sim_suffix)),
        "edge_overlap_simulator": _jaccard(_edges(planned_suffix), _edges(sim_suffix)),
        "route_length_ratio_simulator": ratio,
    }


def _evaluate_records(
    model,
    records: list[EpisodeRecord],
    split: str,
    meta: dict[str, Any],
    node_coords: dict[int, tuple[float, float]],
    router: Router,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    goal_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    simulator_rows: list[dict[str, Any]] = []

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
        for local_idx, (record, comp) in enumerate(zip(records, comps)):
            source_index = args.record_start + local_idx
            prefix_len = _prefix_len(record.path, fraction)
            prefix = record.path[:prefix_len]
            original_suffix = record.path[prefix_len - 1 :]
            intervention_node = prefix[-1]
            original_goal_idx = meta["poi_to_idx"][record.goal_poi]
            normal_logits = comp["goal_logits"].float()
            normal_probs, normal_top5 = _topk_from_logits(normal_logits)
            normal_top1_idx = int(normal_top5[0].item())
            normal_top1_poi = meta["idx_to_poi"][normal_top1_idx]

            for cf_type in CF_TYPES:
                sim = _simulator_counterfactual(
                    record,
                    meta,
                    router,
                    cf_type,
                    intervention_node,
                    source_index,
                    fraction,
                    args.seed,
                )
                for channel_mode in CHANNEL_MODES:
                    desire_cf, belief_cf, _, candidate_category = _edited_channels(
                        comp,
                        record,
                        meta,
                        cf_type,
                        channel_mode,
                    )
                    with torch.no_grad():
                        cf_logits = model.predict_goal(
                            desire_cf.unsqueeze(0).to(device),
                            belief_cf.unsqueeze(0).to(device),
                            comp["z_tau"].unsqueeze(0).to(device),
                        )[0].detach().cpu().float()
                    cf_probs, cf_top5 = _topk_from_logits(cf_logits)
                    cf_top1_idx = int(cf_top5[0].item())
                    cf_top1_poi = meta["idx_to_poi"][cf_top1_idx]
                    cf_goal_node = int(meta["poi_to_node"][cf_top1_poi])
                    planned_suffix = _plan_path(
                        router,
                        intervention_node,
                        cf_goal_node,
                        args.seed + source_index * 7919 + int(round(fraction * 1000)) * 13 + cf_top1_idx,
                    )
                    planned_success = int(bool(planned_suffix))
                    if not planned_suffix:
                        planned_suffix = [intervention_node]
                    planned_path = prefix[:-1] + planned_suffix
                    metrics = _path_metrics(router, original_suffix, planned_suffix)
                    sim_metrics = _sim_path_metrics(router, planned_suffix, sim["simulator_cf_path"])
                    sim_goal_idx = sim["simulator_cf_goal_idx"]
                    sim_goal_int = int(sim_goal_idx) if sim_goal_idx != "" else -1
                    sim_in_top5 = int(sim_goal_int >= 0 and any(int(x.item()) == sim_goal_int for x in cf_top5))
                    sim_top1 = int(sim_goal_int >= 0 and cf_top1_idx == sim_goal_int)
                    sim_prob = float(cf_probs[sim_goal_int].item()) if sim_goal_int >= 0 else float("nan")

                    base = {
                        "split": split,
                        "record_index": source_index,
                        "local_record_index": local_idx,
                        "record_start": args.record_start,
                        "record_limit": args.record_limit,
                        "chunk_label": args.chunk_label,
                        "agent_id": record.agent_id,
                        "config": record.belief_config,
                        "fraction": fraction,
                        "prefix_len": prefix_len,
                        "path_len": len(record.path),
                        "intervention_node": intervention_node,
                        "cf_type": cf_type,
                        "channel_mode": channel_mode,
                        "candidate_category": candidate_category,
                        "original_goal": record.goal_poi,
                        "original_goal_idx": original_goal_idx,
                        "normal_top1_goal": normal_top1_poi,
                        "normal_top1_idx": normal_top1_idx,
                        "cf_top1_goal": cf_top1_poi,
                        "cf_top1_idx": cf_top1_idx,
                        "cf_top5_goals": " ".join(meta["idx_to_poi"][int(i.item())] for i in cf_top5),
                        "goal_changed_vs_original": int(cf_top1_idx != original_goal_idx),
                        "goal_changed_vs_normal": int(cf_top1_idx != normal_top1_idx),
                        "normal_prob_original_goal": float(normal_probs[original_goal_idx].item()),
                        "cf_prob_original_goal": float(cf_probs[original_goal_idx].item()),
                        "cf_prob_top1_goal": float(cf_probs[cf_top1_idx].item()),
                    }
                    goal_rows.append(base)
                    path_rows.append(
                        {
                            **base,
                            "planned_success": planned_success,
                            **metrics,
                            "original_prefix": _join_path(prefix),
                            "original_suffix": _join_path(original_suffix),
                            "planned_suffix": _join_path(planned_suffix),
                            "planned_path": _join_path(planned_path),
                        }
                    )
                    simulator_rows.append(
                        {
                            **base,
                            "planned_success": planned_success,
                            "simulator_success": sim["simulator_success"],
                            "simulator_failure_reason": sim["simulator_failure_reason"],
                            "simulator_cf_goal": sim["simulator_cf_goal"],
                            "simulator_cf_goal_idx": sim["simulator_cf_goal_idx"],
                            "simulator_goal_top1_match": sim_top1,
                            "simulator_goal_top5_match": sim_in_top5,
                            "cf_prob_simulator_goal": sim_prob,
                            **sim_metrics,
                            "simulator_suffix": _join_path(sim["simulator_cf_path"]),
                        }
                    )
    return goal_rows, path_rows, simulator_rows


def _float_value(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _aggregate(path_rows: list[dict[str, Any]], simulator_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sim_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in simulator_rows:
        key = (
            row["split"],
            row["cf_type"],
            row["channel_mode"],
            float(row["fraction"]),
            row["record_index"],
        )
        sim_lookup[key] = row

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        grouped[(row["split"], row["cf_type"], row["channel_mode"], float(row["fraction"]))].append(row)

    out = []
    for (split, cf_type, channel_mode, fraction), rows in sorted(grouped.items()):
        sim_rows = [
            sim_lookup.get((r["split"], r["cf_type"], r["channel_mode"], float(r["fraction"]), r["record_index"]), {})
            for r in rows
        ]
        out.append(
            {
                "split": split,
                "cf_type": cf_type,
                "channel_mode": channel_mode,
                "fraction": fraction,
                "n": len(rows),
                "planned_success_rate": safe_mean([_float_value(r, "planned_success") for r in rows]),
                "goal_change_rate_vs_original": safe_mean([_float_value(r, "goal_changed_vs_original") for r in rows]),
                "goal_change_rate_vs_normal": safe_mean([_float_value(r, "goal_changed_vs_normal") for r in rows]),
                "mean_node_overlap_original": safe_mean([_float_value(r, "node_overlap_original") for r in rows]),
                "mean_edge_overlap_original": safe_mean([_float_value(r, "edge_overlap_original") for r in rows]),
                "mean_route_length_ratio_original": safe_mean([_float_value(r, "route_length_ratio_original") for r in rows]),
                "simulator_success_rate": safe_mean([_float_value(r, "simulator_success") for r in sim_rows if r]),
                "simulator_goal_top1_match": safe_mean([_float_value(r, "simulator_goal_top1_match") for r in sim_rows if r]),
                "simulator_goal_top5_match": safe_mean([_float_value(r, "simulator_goal_top5_match") for r in sim_rows if r]),
                "mean_node_overlap_simulator": safe_mean([_float_value(r, "node_overlap_simulator") for r in sim_rows if r]),
                "mean_edge_overlap_simulator": safe_mean([_float_value(r, "edge_overlap_simulator") for r in sim_rows if r]),
                "mean_route_length_ratio_simulator": safe_mean([_float_value(r, "route_length_ratio_simulator") for r in sim_rows if r]),
            }
        )
    return out


def _summary(goal_rows: list[dict[str, Any]], path_rows: list[dict[str, Any]], simulator_rows: list[dict[str, Any]]) -> str:
    n = len(path_rows)
    planned = sum(1 for row in path_rows if int(row["planned_success"]) == 1)
    changed_original = safe_mean([float(row["goal_changed_vs_original"]) for row in goal_rows])
    changed_normal = safe_mean([float(row["goal_changed_vs_normal"]) for row in goal_rows])
    node_overlap = safe_mean([float(row["node_overlap_original"]) for row in path_rows])
    edge_overlap = safe_mean([float(row["edge_overlap_original"]) for row in path_rows])
    sim_success = safe_mean([float(row["simulator_success"]) for row in simulator_rows])
    sim_top1 = safe_mean([float(row["simulator_goal_top1_match"]) for row in simulator_rows])
    sim_top5 = safe_mean([float(row["simulator_goal_top5_match"]) for row in simulator_rows])
    return "\n".join(
        [
            "# Model-Planned Counterfactual Path Summary",
            "",
            f"- Model-planned path rows: {n}.",
            f"- Successful model-planned routes: {planned}.",
            f"- Goal-change rate vs original goal: {changed_original:.4f}.",
            f"- Goal-change rate vs normal model top-1: {changed_normal:.4f}.",
            f"- Mean node overlap with original future suffix: {node_overlap:.4f}.",
            f"- Mean edge overlap with original future suffix: {edge_overlap:.4f}.",
            f"- Simulator comparison success rate: {sim_success:.4f}.",
            f"- Simulator counterfactual goal top-1 agreement: {sim_top1:.4f}.",
            f"- Simulator counterfactual goal top-5 agreement: {sim_top5:.4f}.",
            "",
            "This diagnostic freezes BDIBottleneck, edits belief/desire channels at an observed prefix, lets the model pick a counterfactual goal, and routes from the intervention node to that model-planned goal.",
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


def _write_outputs(
    out_dir,
    goal_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    simulator_rows: list[dict[str, Any]],
) -> None:
    write_csv(out_dir / "goal_changes.csv", goal_rows)
    write_csv(out_dir / "planned_paths.csv", path_rows)
    write_csv(out_dir / "model_vs_simulator.csv", simulator_rows)
    write_csv(out_dir / "path_metrics_by_fraction.csv", _aggregate(path_rows, simulator_rows))
    write_text(out_dir / "summary.md", _summary(goal_rows, path_rows, simulator_rows))


def _merge_chunks(chunks_dir: str, out_dir) -> None:
    goal_rows = _collect_chunk_rows(chunks_dir, out_dir, "goal_changes.csv")
    path_rows = _collect_chunk_rows(chunks_dir, out_dir, "planned_paths.csv")
    simulator_rows = _collect_chunk_rows(chunks_dir, out_dir, "model_vs_simulator.csv")
    _write_outputs(out_dir, goal_rows, path_rows, simulator_rows)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.fractions = args.fractions[:2]
        if args.max_records <= 0 and args.record_limit <= 0:
            args.record_limit = args.smoke_n

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_chunks:
        _merge_chunks(args.merge_chunks, out_dir)
        print(f"Merged model-planned counterfactual path chunks into {out_dir}")
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

    goal_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    simulator_rows: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        records = _slice_records(records, args)
        split_goal, split_path, split_sim = _evaluate_records(
            model,
            records,
            split,
            meta,
            node_coords,
            router,
            device,
            args,
        )
        goal_rows.extend(split_goal)
        path_rows.extend(split_path)
        simulator_rows.extend(split_sim)

    _write_outputs(out_dir, goal_rows, path_rows, simulator_rows)
    print(f"Model-planned counterfactual path diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
