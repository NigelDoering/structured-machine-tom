"""
scripts/extract_distractor_episodes.py

Extract the "distractor" subset of test episodes for targeted evaluation.

Setting
-------
An agent travels toward a high-preference destination (top-K by desire_vector)
while their path passes within `--threshold` metres of a low-preference POI
(bottom-K by desire_vector).  A model without mental-state awareness should
assign inflated probability to the spatially proximate distractor; the
BDIPredictor should correctly assign higher probability to the true goal.

Filtering conditions (all three must hold for a record to be kept):
  1. goal_poi is among the agent's top-K preferred POIs
  2. At least one bottom-K POI node lies within `--threshold` metres of some
     node on the path (excluding the last `--end-margin` steps)
  3. The distractor POI is different from the goal (always true by construction,
     since top-K and bottom-K are disjoint)

Output
------
  data/splits/distractor_{split}.jsonl      — filtered EpisodeRecords
  data/splits/distractor_{split}_meta.json  — per-record metadata:
      agent_id, episode_idx, belief_config, goal_poi,
      distractor_poi, min_dist_m

Usage
-----
  .venv/bin/python scripts/extract_distractor_episodes.py
  .venv/bin/python scripts/extract_distractor_episodes.py --threshold 75 --top-k 3 --bot-k 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulation.episode import EpisodeRecord


# ── Geometry ───────────────────────────────────────────────────────────

def _meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate planar distance in metres between two (lat, lon) points."""
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * 111_000 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dlat, dlon)


# ── POI helpers ────────────────────────────────────────────────────────

def _build_poi_tables(poi_cfg: dict) -> tuple[list[str], dict[str, int]]:
    """
    Returns
    -------
    all_poi_ids : ordered list of 24 POI id strings — same order as desire_vector
    poi_to_node : {poi_id: osm_node_id}
    """
    all_poi_ids: list[str] = []
    poi_to_node: dict[str, int] = {}
    for cat_data in poi_cfg["categories"].values():
        for poi in cat_data["pois"]:
            all_poi_ids.append(poi["id"])
            poi_to_node[poi["id"]] = poi["node_id"]
    return all_poi_ids, poi_to_node


# ── Core filtering ─────────────────────────────────────────────────────

def _closest_approach(
    path:           list[int],
    target_node:    int,
    node_coords:    dict[int, tuple[float, float]],
    end_margin:     int,
) -> tuple[float, int]:
    """
    Minimum distance (metres) from target_node to any node on the path,
    considering only the first len(path) - end_margin steps.

    Returns (min_dist_m, path_index_of_closest).
    Returns (inf, -1) if target_node has no coordinates.
    """
    if target_node not in node_coords:
        return float("inf"), -1

    t_lat, t_lon = node_coords[target_node]
    cutoff   = max(1, len(path) - end_margin)
    min_dist = float("inf")
    min_idx  = -1

    for i, nid in enumerate(path[:cutoff]):
        coord = node_coords.get(nid)
        if coord is None:
            continue
        d = _meters(coord[0], coord[1], t_lat, t_lon)
        if d < min_dist:
            min_dist = d
            min_idx  = i

    return min_dist, min_idx


def filter_distractor_episodes(
    records:     list[EpisodeRecord],
    all_poi_ids: list[str],
    poi_to_node: dict[str, int],
    node_coords: dict[int, tuple[float, float]],
    top_k:       int   = 5,
    bot_k:       int   = 5,
    threshold_m: float = 100.0,
    end_margin:  int   = 10,
) -> tuple[list[EpisodeRecord], list[dict]]:
    """
    Filter records to the distractor subset and return parallel metadata.

    Returns
    -------
    kept     : filtered EpisodeRecords
    metadata : one dict per kept record with distractor details
    """
    kept:     list[EpisodeRecord] = []
    metadata: list[dict]          = []

    for r in records:
        dv = r.desire_vector   # length-24 float list

        # Rank all 24 POIs by preference (descending)
        ranked   = sorted(range(len(dv)), key=lambda i: dv[i], reverse=True)
        top_pois = {all_poi_ids[i] for i in ranked[:top_k]}
        bot_pois = [all_poi_ids[i] for i in ranked[-bot_k:]]

        # Condition 1: goal is high-preference
        if r.goal_poi not in top_pois:
            continue

        # Condition 2: path passes within threshold_m of a low-preference POI
        # before the final end_margin steps — find the closest such distractor
        best_distractor: str | None = None
        best_dist_m: float          = float("inf")

        for dpoi in bot_pois:
            dnode = poi_to_node.get(dpoi)
            if dnode is None:
                continue
            dist_m, _ = _closest_approach(r.path, dnode, node_coords, end_margin)
            if dist_m <= threshold_m and dist_m < best_dist_m:
                best_dist_m     = dist_m
                best_distractor = dpoi

        if best_distractor is None:
            continue

        # Re-derive the path index of closest approach for the winning distractor
        best_node = poi_to_node[best_distractor]
        _, best_path_idx = _closest_approach(r.path, best_node, node_coords, end_margin)

        kept.append(r)
        metadata.append({
            "agent_id":          r.agent_id,
            "episode_idx":       r.episode_idx,
            "belief_config":     r.belief_config,
            "goal_poi":          r.goal_poi,
            "distractor_poi":    best_distractor,
            "min_dist_m":        round(best_dist_m, 1),
            "closest_path_idx":  best_path_idx,   # truncate path here for eval
            "path_len":          len(r.path),
        })

    return kept, metadata


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract distractor episodes from test splits."
    )
    p.add_argument("--splits-dir",  default="data/splits",          metavar="DIR")
    p.add_argument("--threshold",   type=float, default=100.0,       metavar="M",
                   help="Max metres from path to distractor POI node (default: 100)")
    p.add_argument("--top-k",       type=int,   default=5,           metavar="K",
                   help="High-preference POI rank cutoff (default: 5)")
    p.add_argument("--bot-k",       type=int,   default=5,           metavar="K",
                   help="Low-preference (distractor) POI rank cutoff (default: 5)")
    p.add_argument("--end-margin",  type=int,   default=10,          metavar="N",
                   dest="end_margin",
                   help="Ignore final N path steps when checking proximity (default: 10)")
    p.add_argument("--splits",      nargs="+",
                   default=["test", "test_new_agent"],
                   help="Which splits to filter (default: test test_new_agent)")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    splits_dir = ROOT / args.splits_dir

    # ── POI config ────────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    all_poi_ids, poi_to_node = _build_poi_tables(poi_cfg)

    # ── Graph node coordinates ────────────────────────────────────────
    try:
        import osmnx as ox
        graph_path = ROOT / "data" / "ucsd_walk.graphml"
        print(f"Loading graph from {graph_path} …")
        G = ox.load_graphml(graph_path)
        node_coords: dict[int, tuple[float, float]] = {
            int(nid): (float(data["y"]), float(data["x"]))
            for nid, data in G.nodes(data=True)
        }
        print(f"  {len(node_coords):,} nodes loaded\n")
    except Exception as exc:
        print(f"ERROR loading graph: {exc}")
        sys.exit(1)

    print(f"Settings: threshold={args.threshold}m  top_k={args.top_k}  "
          f"bot_k={args.bot_k}  end_margin={args.end_margin}\n")

    # ── Filter each split ─────────────────────────────────────────────
    for split_name in args.splits:
        src = splits_dir / f"{split_name}.jsonl"
        if not src.exists():
            print(f"  WARNING: {src} not found — skipping.")
            continue

        records: list[EpisodeRecord] = []
        with open(src) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EpisodeRecord.from_json(line))

        kept, metadata = filter_distractor_episodes(
            records, all_poi_ids, poi_to_node, node_coords,
            top_k       = args.top_k,
            bot_k       = args.bot_k,
            threshold_m = args.threshold,
            end_margin  = args.end_margin,
        )

        # Save filtered records
        out_jsonl = splits_dir / f"distractor_{split_name}.jsonl"
        with open(out_jsonl, "w") as f:
            for r in kept:
                f.write(r.to_json() + "\n")

        # Save per-record metadata
        out_meta = splits_dir / f"distractor_{split_name}_meta.json"
        with open(out_meta, "w") as f:
            json.dump(metadata, f, indent=2)

        pct = 100 * len(kept) / len(records) if records else 0.0
        print(f"  {split_name:<20}  {len(records):>5} total  →  "
              f"{len(kept):>5} distractor  ({pct:.1f}%)")
        print(f"    Records  → {out_jsonl}")
        print(f"    Metadata → {out_meta}")

        # Quick distractor distribution summary
        from collections import Counter
        dist_counts = Counter(m["distractor_poi"] for m in metadata)
        print(f"    Top distractors:")
        for poi, cnt in dist_counts.most_common(5):
            print(f"      {poi:<30} {cnt:>4} episodes")
        print()


if __name__ == "__main__":
    main()
