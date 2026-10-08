import json
import numpy as np
from pathlib import Path


CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


class Agent:
    """
    A single simulated agent.

    At instantiation the agent:
      - Loads its character definition (category and node Dirichlet parameters)
      - Samples a concrete preference distribution over desire categories
      - Samples a concrete preference distribution over POIs within each category
      - Initialises a belief vector of 24 ones (all POIs believed open)

    The belief vector is the only mutable state. Everything else is fixed for
    the lifetime of the agent.
    """

    def __init__(self, agent_id: str, character_name: str, seed: int | None = None):
        self.agent_id = agent_id
        self.character_name = character_name
        self.rng = np.random.default_rng(seed)

        chars = _load_json(CONFIG_DIR / "agent_characters.json")
        pois  = _load_json(CONFIG_DIR / "poi_categories.json")

        if character_name not in chars["characters"]:
            raise ValueError(
                f"Unknown character '{character_name}'. "
                f"Available: {list(chars['characters'].keys())}"
            )

        char = chars["characters"][character_name]
        self.description = char["description"]

        # Ordered list of categories and POI ids — stable reference throughout
        self.categories = list(pois["categories"].keys())          # length 6
        self.poi_ids: dict[str, list[str]] = {
            cat: [p["id"] for p in data["pois"]]
            for cat, data in pois["categories"].items()
        }
        self.poi_node_ids: dict[str, list[int]] = {
            cat: [p["node_id"] for p in data["pois"]]
            for cat, data in pois["categories"].items()
        }

        # Flat ordered list of all 24 POI ids — determines belief vector index
        self.all_poi_ids: list[str] = [
            poi_id
            for cat in self.categories
            for poi_id in self.poi_ids[cat]
        ]  # length 24

        # Sample category preference vector from Dirichlet(category_alphas)
        cat_alphas = np.array(
            [char["category_alphas"][c] for c in self.categories], dtype=float
        )
        self.category_prefs: np.ndarray = self.rng.dirichlet(cat_alphas)

        # Sample within-category preference vectors from Dirichlet(node_alphas)
        self.node_prefs: dict[str, np.ndarray] = {}
        for cat in self.categories:
            node_alphas = np.array(char["node_alphas"][cat], dtype=float)
            self.node_prefs[cat] = self.rng.dirichlet(node_alphas)

        # Belief vector: 1 = believed open, 0 = falsely believed closed
        # Indexed in the same order as self.all_poi_ids
        self.belief: np.ndarray = np.ones(len(self.all_poi_ids), dtype=np.int8)

        # Active belief configuration — updated by set_config()
        self.current_config: str = "B0"
        self.current_false_beliefs: list[str] = []

    # ------------------------------------------------------------------
    # Belief management
    # ------------------------------------------------------------------

    def set_config(self, config_name: str, false_poi_ids: list[str]) -> None:
        """
        Apply a named belief configuration atomically.

        Resets all beliefs to 1, then marks each POI in false_poi_ids as 0.
        Records the config name and false-belief list on the agent so that
        episode records can be self-contained.
        """
        self.reset_belief()
        for poi_id in false_poi_ids:
            self.set_belief_by_poi(poi_id, 0)
        self.current_config = config_name
        self.current_false_beliefs = list(false_poi_ids)

    def set_belief(self, belief: list[int] | np.ndarray) -> None:
        """Replace the full belief vector. Must be length 24, values in {0, 1}."""
        b = np.asarray(belief, dtype=np.int8)
        if b.shape != (len(self.all_poi_ids),):
            raise ValueError(f"Belief vector must have length {len(self.all_poi_ids)}, got {b.shape}")
        if not np.all((b == 0) | (b == 1)):
            raise ValueError("Belief vector must contain only 0 or 1.")
        self.belief = b.copy()

    def set_belief_by_poi(self, poi_id: str, value: int) -> None:
        """Set a single POI's belief entry by POI id."""
        idx = self.all_poi_ids.index(poi_id)
        self.belief[idx] = value

    def reset_belief(self) -> None:
        """Restore all beliefs to 1 (all POIs believed open)."""
        self.belief[:] = 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_desire(self) -> str:
        """Sample a desire category weighted by the agent's category preferences."""
        return str(self.rng.choice(self.categories, p=self.category_prefs))

    def sample_goal(self, desire: str) -> str | None:
        """
        Sample a specific POI id for the given desire category, considering
        only POIs the agent currently believes are open.

        Returns None if the agent believes all POIs in this category are closed.
        """
        poi_ids   = self.poi_ids[desire]
        prefs     = self.node_prefs[desire]
        beliefs   = np.array(
            [self.belief[self.all_poi_ids.index(pid)] for pid in poi_ids]
        )

        available_mask = beliefs == 1
        if not available_mask.any():
            return None

        # Re-weight preferences over believed-open POIs and renormalise
        masked_prefs = prefs * available_mask
        masked_prefs = masked_prefs / masked_prefs.sum()

        return str(self.rng.choice(poi_ids, p=masked_prefs))

    def goal_node_id(self, poi_id: str) -> int:
        """Return the OSM node_id for a given POI id."""
        for cat in self.categories:
            if poi_id in self.poi_ids[cat]:
                idx = self.poi_ids[cat].index(poi_id)
                return self.poi_node_ids[cat][idx]
        raise ValueError(f"POI id '{poi_id}' not found.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def poi_preference_vector(self) -> np.ndarray:
        """
        24-dimensional joint preference distribution over all POIs.

        P(poi_i) = P(category_of_poi_i) × P(poi_i | category_of_poi_i)

        This is the agent's stable desire signal — independent of belief state
        and constant for the lifetime of the agent. Indexed in the same order
        as self.all_poi_ids. Sums to 1.
        """
        prefs = np.zeros(len(self.all_poi_ids), dtype=np.float64)
        poi_idx = 0
        for cat_idx, cat in enumerate(self.categories):
            cat_prob = float(self.category_prefs[cat_idx])
            for node_prob in self.node_prefs[cat]:
                prefs[poi_idx] = cat_prob * float(node_prob)
                poi_idx += 1
        return prefs / prefs.sum()

    def belief_dict(self) -> dict[str, int]:
        """Return belief vector as {poi_id: 0|1}."""
        return dict(zip(self.all_poi_ids, self.belief.tolist()))

    def false_beliefs(self) -> list[str]:
        """Return list of POI ids the agent falsely believes are closed."""
        return [pid for pid, b in self.belief_dict().items() if b == 0]

    def __repr__(self) -> str:
        false = self.false_beliefs()
        belief_str = "all correct" if not false else f"false on {false}"
        return (
            f"Agent(id={self.agent_id!r}, character={self.character_name!r}, "
            f"beliefs={belief_str})"
        )
