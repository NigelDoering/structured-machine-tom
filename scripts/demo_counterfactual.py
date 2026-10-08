"""
Minimal example: attribute mental states with the released SMToM model and
run one belief and one desire counterfactual.

Usage:
    .venv/bin/python scripts/demo_counterfactual.py                    # first test episode
    .venv/bin/python scripts/demo_counterfactual.py --episode 17 --fraction 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import osmnx as ox
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.bdi_bottleneck import BDIBottleneckPredictor
from model.dataset import build_poi_index, node_coords_from_graph
from model.meta_dataset import MetaEpisodeDataset
from simulation.episode import EpisodeRecord


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/smtom-bdibottleneck.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--episode", type=int, default=0, help="Index of the episode within the split")
    p.add_argument("--fraction", type=float, default=0.5, help="Fraction of the current path observed")
    return p.parse_args()


def top(probs: torch.Tensor, names: list[str], k: int = 3) -> str:
    vals, idx = probs.topk(k)
    return ", ".join(f"{names[i]} {v:.2f}" for v, i in zip(vals.tolist(), idx.tolist()))


def main() -> None:
    args = parse_args()

    poi_cfg = json.loads((ROOT / "config" / "poi_categories.json").read_text())
    build_poi_index(poi_cfg)
    names = [p["id"] for cat in poi_cfg["categories"].values() for p in cat["pois"]]
    poi_to_idx = {name: i for i, name in enumerate(names)}
    category = {p["id"]: c for c, cat in poi_cfg["categories"].items() for p in cat["pois"]}

    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu")
    model = BDIBottleneckPredictor(num_pois=len(names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # One episode plus its same-agent context (K=10 desire, K_recent=5 recent trajectories).
    lines = (ROOT / "data" / "splits" / f"{args.split}.jsonl").read_text().splitlines()
    records = [EpisodeRecord.from_json(line) for line in lines if line.strip()]
    node_coords = node_coords_from_graph(ox.load_graphml(ROOT / "data" / "ucsd_walk.graphml"))
    ds = MetaEpisodeDataset(records, node_coords, poi_to_idx)
    batch = MetaEpisodeDataset.collate_fn([ds[args.episode]])

    # Observe only the first ceil(fraction * T) steps of the current trajectory.
    n_obs = max(1, math.ceil(args.fraction * int(batch["current_lengths"][0])))
    batch["current_mask"][:, n_obs:] = 0

    with torch.no_grad():
        desire, belief, z_tau = model.encode_components(
            batch["desire_coords"], batch["desire_mask"],
            batch["recent_coords"], batch["recent_mask"],
            batch["current_coords"], batch["current_mask"],
        )
        goal = model.predict_goal(desire, belief, z_tau).softmax(-1)[0]

        r = records[args.episode]
        print(f"Agent {r.agent_id} ({r.character}), true goal: {r.goal_poi}, "
              f"observed {n_obs}/{len(r.path)} steps")
        print(f"  inferred desires : {top(desire[0], names)}")
        closed = [names[i] for i in (belief[0] < 0.5).nonzero().flatten().tolist()]
        print(f"  believed closed  : {', '.join(closed) or 'none'}")
        print(f"  predicted goals  : {top(goal, names)}")

        # Belief counterfactual: the agent now believes its most likely goal is closed.
        g = int(goal.argmax())
        belief_cf = belief.clone()
        belief_cf[0, g] = 0.0
        goal_cf = model.predict_goal(desire, belief_cf, z_tau).softmax(-1)[0]
        print(f"\nIf the agent believed {names[g]} were closed:")
        print(f"  {'P(' + names[g] + ')':<17}: {goal[g]:.2f} -> {goal_cf[g]:.2f}")
        print(f"  predicted goals  : {top(goal_cf, names)}")

        # Desire counterfactual: replace inferred desires with a uniform get_food prototype.
        food = torch.tensor([category[n] == "get_food" for n in names])
        desire_cf = (food.float() / food.sum()).unsqueeze(0)
        goal_cf = model.predict_goal(desire_cf, belief, z_tau).softmax(-1)[0]
        print("\nIf the agent wanted food instead:")
        print(f"  P(any food POI)  : {goal[food].sum():.2f} -> {goal_cf[food].sum():.2f}")
        print(f"  predicted goals  : {top(goal_cf, names)}")

    print("\nBelief edits move probability less than desire edits, and the observed path")
    print("still constrains the goal; see Sections 6.2-6.3 of the paper.")


if __name__ == "__main__":
    main()
