"""
scripts/run_simulation.py

Run the full BDI ToM simulation and write all dataset splits to disk.

Usage
-----
  .venv/bin/python scripts/run_simulation.py

Optional flags
--------------
    --agents-per-char 3          Agents per character archetype (default: 3 => 15 total)
    --standard-total 25          Pooled episodes per (agent, training-config) cell
    --train-per-cell 17          Per-cell records assigned to train
    --val-per-cell 4             Per-cell records assigned to val
    --test-per-cell 4            Per-cell records assigned to test
    --new-test-per-cell 4        Episodes per (new-agent, training-config) cell
    --bh-per-cell 16             Episodes per (agent, held-out-config) cell for BH splits
    --seed 9999                  RNG seed for new agents
    --out data/splits            Output directory for JSONL files

Output
------
  <out>/train.jsonl
  <out>/val.jsonl
  <out>/test.jsonl
  <out>/test_new_agent.jsonl
    <out>/test_bh.jsonl
    <out>/test_new_agent_bh.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulation.agent import Agent
from simulation.router import Router
from simulation.runner import SimulationRunner


# ── Candidate paths for the campus walk graph ─────────────────────────
GRAPH_CANDIDATES = [
    ROOT / "data" / "ucsd_walk.graphml",
    ROOT / "bdi_tom" / "data" / "raw" / "ucsd_walk.graphml",
    ROOT / "data" / "raw" / "ucsd_walk.graphml",
]


def load_or_download_graph(save_to: Path = GRAPH_CANDIDATES[0]):
    try:
        import osmnx as ox
    except ImportError:
        print("ERROR: osmnx not installed. Run: .venv/bin/pip install osmnx")
        sys.exit(1)

    for p in GRAPH_CANDIDATES:
        if p.exists():
            print(f"Loading graph from {p}")
            return ox.load_graphml(p)

    print(f"Graph not found — downloading UCSD walk graph → {save_to}")
    G = ox.graph_from_bbox(
        bbox=(-117.253, 32.867, -117.220, 32.895),
        network_type="walk",
    )
    save_to.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, save_to)
    return G


def build_agents(char_cfg: dict, agents_per_char: int, seed_base: int = 42) -> list[Agent]:
    """Create `agents_per_char` agents for every character archetype."""
    agents = []
    for char in char_cfg["characters"]:
        for i in range(agents_per_char):
            agents.append(
                Agent(
                    agent_id=f"{char}_{i}",
                    character_name=char,
                    seed=seed_base + i,
                )
            )
    return agents


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the BDI ToM simulation and write stratified dataset splits to JSONL."
    )
    p.add_argument("--configs", nargs="+", default=None,
                   metavar="CONFIG",
                   help="Training belief config names to run (default: all training configs)")
    p.add_argument("--heldout-configs", nargs="+", default=None,
                   metavar="CONFIG",
                   help="Held-out belief configs for BH evaluation (default: all held-out configs)")
    p.add_argument("--standard-total", type=int, default=25, metavar="N",
                   dest="standard_total",
                   help="Pooled episodes per (agent, training-config) cell before partition (default: 25)")
    p.add_argument("--train-per-cell", type=int, default=17, metavar="N",
                   dest="train_per_cell",
                   help="Per-cell episodes assigned to train (default: 17)")
    p.add_argument("--val-per-cell", type=int, default=4, metavar="N",
                   dest="val_per_cell",
                   help="Per-cell episodes assigned to val (default: 4)")
    p.add_argument("--test-per-cell", type=int, default=4, metavar="N",
                   dest="test_per_cell",
                   help="Per-cell episodes assigned to test (default: 4)")
    p.add_argument("--new-test-per-cell", type=int, default=4, metavar="N",
                   dest="new_test_per_cell",
                   help="Episodes per (new-agent, training-config) cell for test_new_agent (default: 4)")
    p.add_argument("--bh-per-cell", type=int, default=16, metavar="N",
                   dest="bh_per_cell",
                   help="Episodes per (agent, held-out-config) cell for BH splits (default: 16)")
    p.add_argument("--seed",   type=int, default=9999, metavar="SEED",
                   help="RNG seed for new test agents (default: 9999)")
    p.add_argument("--out",    type=str, default="data/splits", metavar="DIR",
                   help="Output directory for JSONL files (default: data/splits)")
    p.add_argument("--belief-config", type=str, default="config/belief_configs.json",
                   metavar="JSON",
                   help="Belief configuration JSON (default: config/belief_configs.json)")
    p.add_argument("--agents-per-char", type=int, default=3, metavar="N",
                   dest="agents_per_char",
                   help="Agents to create per character archetype (default: 3)")
    p.add_argument("--router-k", type=int, default=5, metavar="K",
                   dest="router_k",
                   help="Candidate shortest paths sampled by the router (default: 5)")
    p.add_argument("--router-temperature", type=float, default=100.0, metavar="TAU",
                   dest="router_temperature",
                   help="Boltzmann temperature for route sampling in metres (default: 100.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out

    # ── Load configs ──────────────────────────────────────────────────
    with open(ROOT / "config" / "poi_categories.json") as f:
        poi_cfg = json.load(f)
    with open(ROOT / "config" / "agent_characters.json") as f:
        char_cfg = json.load(f)
    belief_cfg_path = ROOT / args.belief_config
    with open(belief_cfg_path) as f:
        belief_cfg = json.load(f)

    configs = args.configs or list(belief_cfg["training"].keys())
    heldout_configs = args.heldout_configs or list(belief_cfg.get("held_out", {}).keys())

    if args.train_per_cell + args.val_per_cell + args.test_per_cell > args.standard_total:
        raise ValueError(
            "Invalid settings: train_per_cell + val_per_cell + test_per_cell "
            "must be <= standard_total"
        )

    print("=" * 58)
    print("  BDI ToM Simulation")
    print("=" * 58)
    print(f"  Training cfgs           : {configs}")
    print(f"  Held-out cfgs (BH)      : {heldout_configs}")
    print(f"  Pooled train-cell eps   : {args.standard_total}")
    print(
        "  Per-cell partition      : "
        f"train={args.train_per_cell}, val={args.val_per_cell}, test={args.test_per_cell}"
    )
    print(f"  New-agent test per-cell : {args.new_test_per_cell}  (seed={args.seed})")
    print(f"  BH per-cell episodes    : {args.bh_per_cell}")
    print(f"  Agents/char    : {args.agents_per_char}")
    print(f"  Belief config  : {belief_cfg_path}")
    print(f"  Router k/tau   : {args.router_k} / {args.router_temperature}")
    print(f"  Output dir     : {out_dir}")
    print("=" * 58)

    # ── Build agents ──────────────────────────────────────────────────
    agents = build_agents(char_cfg, args.agents_per_char)
    print(f"\nCreated {len(agents)} agents:")
    for a in agents:
        print(f"  {a}")

    # ── Load graph + router ───────────────────────────────────────────
    print("\nLoading graph...")
    t0 = time.time()
    graph_path = next((p for p in GRAPH_CANDIDATES if p.exists()), GRAPH_CANDIDATES[0])
    if not graph_path.exists():
        load_or_download_graph(graph_path)   # ensures file exists
    else:
        print(f"Loading graph from {graph_path}")
    router = Router(graph_path=graph_path, k=args.router_k, temperature=args.router_temperature)
    print(f"Router ready  ({time.time() - t0:.1f}s)")

    # ── Run simulation ────────────────────────────────────────────────
    runner = SimulationRunner(
        agents     = agents,
        router     = router,
        belief_cfg = belief_cfg,
        out_dir    = out_dir,
        verbose    = True,
    )

    t1 = time.time()
    runner.run_stratified(
        standard_configs           = configs,
        standard_total_per_cell    = args.standard_total,
        train_per_cell             = args.train_per_cell,
        val_per_cell               = args.val_per_cell,
        test_per_cell              = args.test_per_cell,
        new_agent_standard_per_cell= args.new_test_per_cell,
        heldout_configs            = heldout_configs,
        bh_per_cell                = args.bh_per_cell,
        new_agent_seed             = args.seed,
    )
    print(f"\nSimulation complete  ({time.time() - t1:.1f}s)")


if __name__ == "__main__":
    main()
