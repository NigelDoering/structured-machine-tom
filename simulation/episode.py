"""
simulation/episode.py

EpisodeRecord dataclass and the simulate_episode() function.

An episode is one BDI decision cycle:
    sample desire → filter by beliefs → sample goal → route

The record captures everything needed for model training: the agent's
full identity, belief state, sampled mental states, and the resulting
trajectory on the graph.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class EpisodeRecord:
    """
    One labeled training trajectory.

    Fields
    ------
    agent_id         : unique agent identifier
    character        : character archetype name
    split            : dataset split — "train", "val", "test", or "test_new_agent"
    belief_config    : name of the active belief configuration (e.g. "B1")
    belief_vector    : 24-length binary list — 1 = believed open, 0 = believed closed
    false_beliefs    : POI ids the agent falsely believes are closed
    desire_vector    : 24-length float list — joint POI preference distribution
                       P(poi_i) = P(category_i) × P(poi_i | category_i).
                       Sums to 1. Stable for the agent's lifetime; independent
                       of belief state. Ground-truth supervision signal for
                       desire attribution.
    desire           : sampled desire category for this episode (a single draw
                       from the desire_vector marginal over categories)
    goal_poi         : destination POI id
    goal_node        : OSM node id of the destination
    start_node       : OSM node id where the episode begins
    path             : ordered list of OSM node ids from start to goal
    episode_idx      : index of this episode within the current (agent, config) batch
    """
    agent_id:       str
    character:      str
    split:          str              # "train" | "val" | "test" | "test_new_agent"
    belief_config:  str
    belief_vector:  list[int]        # length 24, binary
    false_beliefs:  list[str]
    desire_vector:  list[float]      # length 24, sums to 1
    desire:         str
    goal_poi:       str
    goal_node:      int
    start_node:     int
    path:           list[int]
    episode_idx:    int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeRecord":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "EpisodeRecord":
        return cls.from_dict(json.loads(s))


def simulate_episode(
    agent,
    router,
    start_node: int,
    episode_idx: int,
    split: str = "",
    max_retries: int = 15,
) -> EpisodeRecord | None:
    """
    Run one BDI episode and return a fully-labelled EpisodeRecord.

    Uses the agent's current belief state (set via agent.set_config() before
    calling this function). Returns None only if routing fails on every retry,
    which should be rare on a well-connected graph.

    Parameters
    ----------
    agent        : Agent instance with beliefs already set
    router       : Router instance for path generation
    start_node   : OSM node id where this episode begins
    episode_idx  : position of this episode in the current (agent, config) batch
    split        : dataset split label ("train", "val", "test", "test_new_agent")
    max_retries  : how many (desire, goal) resamples to attempt before giving up
    """
    for _ in range(max_retries):
        desire   = agent.sample_desire()
        goal_poi = agent.sample_goal(desire)

        if goal_poi is None:
            # All POIs in this category are believed closed — resample desire
            continue

        goal_node = agent.goal_node_id(goal_poi)

        if goal_node == start_node:
            continue

        path = router.route(start_node, goal_node, rng=agent.rng)
        if not path:
            continue

        return EpisodeRecord(
            agent_id      = agent.agent_id,
            character     = agent.character_name,
            split         = split,
            belief_config = agent.current_config,
            belief_vector = agent.belief.tolist(),
            false_beliefs = list(agent.current_false_beliefs),
            desire_vector = agent.poi_preference_vector().tolist(),
            desire        = desire,
            goal_poi      = goal_poi,
            goal_node     = goal_node,
            start_node    = start_node,
            path          = path,
            episode_idx   = episode_idx,
        )

    return None
