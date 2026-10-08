"""
simulation/runner.py

SimulationRunner — orchestrates the full train / val / test / test_new_agent
pipeline and serialises each split to its own JSONL file.

Split design
------------
  train           : existing agents, 25 episodes per (agent, config)
  val             : existing agents, 10 episodes per (agent, config)
  test            : existing agents, 10 episodes per (agent, config)
  test_new_agent  : 5 fresh agents (one per character, different Dirichlet
                    samples), 15 episodes per (agent, config)

Each split starts every agent from their home node independently — no
cross-split chaining. Episodes within a split are chained (end of episode N
becomes start of episode N+1) to produce realistic sequential trajectories.

The test_new_agent split tests generalisation to unseen agents: agents with
the same character archetypes but different preference distributions, produced
by using a different RNG seed at construction time.

Usage
-----
from simulation.runner import SimulationRunner

runner = SimulationRunner(
    agents     = agents,        # list[Agent] — used for train/val/test
    router     = router,        # Router
    belief_cfg = belief_cfg,    # parsed belief_configs.json
    out_dir    = "data/splits",
)
all_records = runner.run_all(configs=["B0", "B1", "B2", "B3", "B4"])
# → data/splits/train.jsonl
# → data/splits/val.jsonl
# → data/splits/test.jsonl
# → data/splits/test_new_agent.jsonl
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from collections import defaultdict

import numpy as np

from simulation.agent import Agent
from simulation.episode import EpisodeRecord, simulate_episode


# Default episode counts per split
_DEFAULTS = {
    "train":          25,
    "val":            10,
    "test":           10,
    "test_new_agent": 15,
}


class SimulationRunner:
    """
    Generates all dataset splits for the BDI ToM simulation.

    Parameters
    ----------
    agents      : Agents used for train / val / test splits.
    router      : Shared Router instance (graph loaded once — expensive).
    belief_cfg  : Parsed belief_configs.json dict. Must have a "training" key;
                  "held_out" is optional.
    out_dir     : If given, each split is saved to {out_dir}/{split}.jsonl
                  automatically after run_all().
    verbose     : Print per-agent, per-config episode counts while running.
    """

    def __init__(
        self,
        agents: list,
        router,
        belief_cfg: dict,
        out_dir: str | Path | None = None,
        verbose: bool = True,
    ):
        self.agents     = list(agents)
        self.router     = router
        self.belief_cfg = belief_cfg
        self.out_dir    = Path(out_dir) if out_dir else None
        self.verbose    = verbose
        self._splits: dict[str, list[EpisodeRecord]] = {}

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run_all(
        self,
        configs: list[str] | None = None,
        train_episodes: int  = _DEFAULTS["train"],
        val_episodes: int    = _DEFAULTS["val"],
        test_episodes: int   = _DEFAULTS["test"],
        new_agent_episodes: int = _DEFAULTS["test_new_agent"],
        new_agent_seed: int  = 9999,
    ) -> dict[str, list[EpisodeRecord]]:
        """
        Run the full pipeline: train, val, test, and test_new_agent splits.

        Parameters
        ----------
        configs            : Belief config names to run. Defaults to all
                             training configs in belief_configs.json.
        train_episodes     : Episodes per (agent, config) for the train split.
        val_episodes       : Episodes per (agent, config) for the val split.
        test_episodes      : Episodes per (agent, config) for the test split.
        new_agent_episodes : Episodes per (new_agent, config) for the
                             test_new_agent split.
        new_agent_seed     : RNG seed for the fresh test agents. Should differ
                             from the seed used to construct self.agents.

        Returns
        -------
        dict mapping split name → list[EpisodeRecord]
        """
        if configs is None:
            configs = list(self.belief_cfg["training"].keys())

        self._splits = {}

        # ── Existing-agent splits ──────────────────────────────────────
        for split_name, n_eps in [
            ("train", train_episodes),
            ("val",   val_episodes),
            ("test",  test_episodes),
        ]:
            if self.verbose:
                print(f"\n{'='*55}")
                print(f"  Split: {split_name}  ({n_eps} episodes × {len(configs)} configs × {len(self.agents)} agents)")
                print(f"{'='*55}")
            records = self._run_agents(self.agents, configs, n_eps, split_name)
            self._splits[split_name] = records
            if self.out_dir:
                self.save(records, self.out_dir / f"{split_name}.jsonl")

        # ── New-agent generalisation split ────────────────────────────
        new_agents = self._make_new_agents(new_agent_seed)
        if self.verbose:
            print(f"\n{'='*55}")
            print(f"  Split: test_new_agent  ({new_agent_episodes} episodes × {len(configs)} configs × {len(new_agents)} fresh agents)")
            print(f"  New-agent seed: {new_agent_seed}")
            print(f"{'='*55}")
        new_records = self._run_agents(new_agents, configs, new_agent_episodes, "test_new_agent")
        self._splits["test_new_agent"] = new_records
        if self.out_dir:
            self.save(new_records, self.out_dir / "test_new_agent.jsonl")

        if self.verbose:
            self._print_summary()

        return self._splits

    def run_stratified(
        self,
        standard_configs: list[str],
        standard_total_per_cell: int,
        train_per_cell: int,
        val_per_cell: int,
        test_per_cell: int,
        new_agent_standard_per_cell: int,
        heldout_configs: list[str] | None = None,
        bh_per_cell: int = 0,
        new_agent_seed: int = 9999,
    ) -> dict[str, list[EpisodeRecord]]:
        """
        Generate splits using fixed per-(agent, config) episode counts.

        Standard agents/configs are generated once at `standard_total_per_cell`
        and partitioned per cell into train/val/test. New-agent and BH splits
        are generated independently with fixed per-cell counts.
        """
        if train_per_cell + val_per_cell + test_per_cell > standard_total_per_cell:
            raise ValueError(
                "Invalid per-cell split counts: train+val+test exceeds standard_total_per_cell"
            )

        self._splits = {}

        if self.verbose:
            print(f"\n{'='*55}")
            print(
                "  Stratified standard pool: "
                f"{standard_total_per_cell} episodes × {len(standard_configs)} configs × {len(self.agents)} agents"
            )
            print(
                "  Per-cell split: "
                f"train={train_per_cell}, val={val_per_cell}, test={test_per_cell}"
            )
            print(f"{'='*55}")

        standard_pool = self._run_agents(
            self.agents,
            standard_configs,
            standard_total_per_cell,
            "standard_pool",
        )
        standard_splits = self._partition_records_per_cell(
            standard_pool,
            {
                "train": train_per_cell,
                "val": val_per_cell,
                "test": test_per_cell,
            },
        )
        self._splits.update(standard_splits)

        new_agents = self._make_new_agents(new_agent_seed)
        if self.verbose:
            print(f"\n{'='*55}")
            print(
                "  Split: test_new_agent  "
                f"({new_agent_standard_per_cell} episodes × {len(standard_configs)} configs × {len(new_agents)} fresh agents)"
            )
            print(f"  New-agent seed: {new_agent_seed}")
            print(f"{'='*55}")

        new_standard = self._run_agents(
            new_agents,
            standard_configs,
            new_agent_standard_per_cell,
            "test_new_agent",
        )
        self._splits["test_new_agent"] = new_standard

        if heldout_configs and bh_per_cell > 0:
            if self.verbose:
                print(f"\n{'='*55}")
                print(
                    "  Split: test_bh  "
                    f"({bh_per_cell} episodes × {len(heldout_configs)} held-out configs × {len(self.agents)} agents)"
                )
                print(f"{'='*55}")
            bh_records = self._run_agents(self.agents, heldout_configs, bh_per_cell, "test_bh")
            self._splits["test_bh"] = bh_records

            if self.verbose:
                print(f"\n{'='*55}")
                print(
                    "  Split: test_new_agent_bh  "
                    f"({bh_per_cell} episodes × {len(heldout_configs)} held-out configs × {len(new_agents)} fresh agents)"
                )
                print(f"  New-agent seed: {new_agent_seed}")
                print(f"{'='*55}")
            new_bh_records = self._run_agents(new_agents, heldout_configs, bh_per_cell, "test_new_agent_bh")
            self._splits["test_new_agent_bh"] = new_bh_records

        if self.out_dir:
            for split_name, records in self._splits.items():
                self.save(records, self.out_dir / f"{split_name}.jsonl")

        if self.verbose:
            self._print_summary()

        return self._splits

    # ------------------------------------------------------------------
    # Convenience: single split (e.g. for quick testing in the notebook)
    # ------------------------------------------------------------------

    def run(
        self,
        configs: list[str] | None = None,
        n_episodes: int = _DEFAULTS["train"],
        split_name: str = "train",
        out_path: str | Path | None = None,
    ) -> list[EpisodeRecord]:
        """
        Run a single split on self.agents and return its records.

        Useful for quick ad-hoc runs or testing without the full pipeline.
        """
        if configs is None:
            configs = list(self.belief_cfg["training"].keys())

        if self.verbose:
            print(f"Running split '{split_name}': {n_episodes} eps × {len(configs)} configs × {len(self.agents)} agents")

        records = self._run_agents(self.agents, configs, n_episodes, split_name)
        self._splits[split_name] = records

        save_to = Path(out_path) if out_path else (self.out_dir / f"{split_name}.jsonl" if self.out_dir else None)
        if save_to:
            self.save(records, save_to)

        return records

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save(records: list[EpisodeRecord], path: str | Path) -> None:
        """Write a list of EpisodeRecords to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(r.to_json() + "\n")
        print(f"  Saved {len(records):>5} records → {path}")

    @staticmethod
    def load(path: str | Path) -> list[EpisodeRecord]:
        """Load EpisodeRecords from a JSONL file."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EpisodeRecord.from_json(line))
        return records

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def splits(self) -> dict[str, list[EpisodeRecord]]:
        """All generated split records, keyed by split name."""
        return {k: list(v) for k, v in self._splits.items()}

    def get_split(self, name: str) -> list[EpisodeRecord]:
        """Return records for a specific split."""
        if name not in self._splits:
            raise KeyError(f"Split '{name}' not found. Available: {list(self._splits.keys())}")
        return list(self._splits[name])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_agents(
        self,
        agents: list,
        configs: list[str],
        n_episodes: int,
        split_name: str,
    ) -> list[EpisodeRecord]:
        """
        Core loop: for each agent, for each config, run n_episodes chained
        episodes starting from the agent's home node. Each split restarts
        from home independently so there is no cross-split information leakage.

        Config order is shuffled independently per agent so that at any point
        in the flattened episode sequence, different agents are experiencing
        different belief configurations.  This creates the inter-agent belief
        variation that the Belief head needs to learn from: K_recent past
        trajectories from agent A reflect config C_A, while those from agent B
        reflect a different config C_B.  Without this, all agents share the
        same config at the same temporal position and z_recent carries no
        agent-discriminative belief signal.
        """
        records: list[EpisodeRecord] = []

        for agent in agents:
            if self.verbose:
                print(f"\n  {agent.agent_id}  ({agent.character_name})")
            home = self._home_node(agent)

            # Shuffle config order per agent using a stable per-agent seed so
            # runs are reproducible while each agent gets a unique ordering.
            # crc32 rather than hash(): str hashing is salted per process.
            agent_rng    = np.random.default_rng(zlib.crc32(agent.agent_id.encode()))
            agent_configs = agent_rng.permutation(configs).tolist()

            for config_name in agent_configs:
                false_pois = self._resolve_config(config_name)
                agent.set_config(config_name, false_pois)

                start      = home
                ep_count   = 0
                skip_count = 0

                max_attempts = max(n_episodes, n_episodes * 5)
                attempts = 0
                while ep_count < n_episodes and attempts < max_attempts:
                    record = simulate_episode(
                        agent, self.router, start,
                        episode_idx=ep_count,
                        split=split_name,
                    )
                    attempts += 1
                    if record is None:
                        skip_count += 1
                        continue
                    records.append(record)
                    start    = record.goal_node
                    ep_count += 1

                if self.verbose:
                    skips = f"  ({skip_count} skipped)" if skip_count else ""
                    print(f"    {config_name}: {ep_count:>3} episodes{skips}")

        return records

    def _partition_records_per_cell(
        self,
        records: list[EpisodeRecord],
        per_cell_counts: dict[str, int],
    ) -> dict[str, list[EpisodeRecord]]:
        """Split a pooled record list into named splits with fixed counts per (agent, config) cell."""
        by_cell: dict[tuple[str, str], list[EpisodeRecord]] = defaultdict(list)
        for r in records:
            by_cell[(r.agent_id, r.belief_config)].append(r)

        total_needed = sum(per_cell_counts.values())
        out: dict[str, list[EpisodeRecord]] = {name: [] for name in per_cell_counts}

        for (agent_id, belief_config), cell_records in by_cell.items():
            # Keep deterministic order within each (agent, config) cell.
            cell_records = sorted(cell_records, key=lambda x: x.episode_idx)
            if len(cell_records) < total_needed:
                raise RuntimeError(
                    "Insufficient episodes in pooled cell "
                    f"({agent_id}, {belief_config}): have {len(cell_records)}, need {total_needed}."
                )

            cursor = 0
            for split_name, n_take in per_cell_counts.items():
                for local_idx, src in enumerate(cell_records[cursor: cursor + n_take]):
                    d = src.to_dict()
                    d["split"] = split_name
                    d["episode_idx"] = local_idx
                    out[split_name].append(EpisodeRecord.from_dict(d))
                cursor += n_take

        return out

    def _make_new_agents(self, seed: int) -> list[Agent]:
        """
        Create fresh agents per character matching the counts in self.agents.

        The new agents use deterministic per-agent seed offsets from `seed`
        so their Dirichlet samples — and therefore their preference
        distributions — differ from the original agents while remaining
        reproducible. Agent IDs are prefixed with "new_".
        """
        by_character: dict[str, int] = defaultdict(int)
        for agent in self.agents:
            by_character[agent.character_name] += 1

        new_agents: list[Agent] = []
        seed_offset = 0
        for char, count in sorted(by_character.items()):
            for i in range(count):
                new_agents.append(
                    Agent(
                        agent_id=f"new_{char}_{i}",
                        character_name=char,
                        seed=seed + seed_offset,
                    )
                )
                seed_offset += 1
        return new_agents

    def _resolve_config(self, config_name: str) -> list[str]:
        """Return the false-belief POI list for a named config."""
        for section in ("training", "held_out"):
            if config_name in self.belief_cfg.get(section, {}):
                return list(self.belief_cfg[section][config_name]["false_beliefs"])
        available = (
            list(self.belief_cfg.get("training", {}).keys())
            + list(self.belief_cfg.get("held_out", {}).keys())
        )
        raise ValueError(f"Unknown belief config '{config_name}'. Available: {available}")

    def _home_node(self, agent) -> int:
        """Starting node = most-preferred rest_home POI."""
        poi_ids = agent.poi_ids["rest_home"]
        prefs   = agent.node_prefs["rest_home"]
        best    = poi_ids[int(np.argmax(prefs))]
        return agent.goal_node_id(best)

    def _print_summary(self) -> None:
        """Print a compact record-count table across all splits."""
        print(f"\n{'─'*40}")
        print(f"  {'Split':<20} {'Records':>8}")
        print(f"{'─'*40}")
        total = 0
        for name, records in self._splits.items():
            print(f"  {name:<20} {len(records):>8}")
            total += len(records)
        print(f"{'─'*40}")
        print(f"  {'TOTAL':<20} {total:>8}")
