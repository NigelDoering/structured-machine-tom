"""
model/dataset.py

PyTorch Dataset and DataLoader utilities for the BDI ToM simulation data.

Variable-length trajectories
-----------------------------
Transformer models handle variable-length sequences natively via attention
masking.  Sequences are padded to the longest in each batch (dynamic padding)
and an attention mask is returned alongside the padded tensor.

Attention mask convention (HuggingFace-style):
    1  →  real token   (attend)
    0  →  padding      (ignore)

Node features
-------------
Each waypoint in a path is represented as its normalised (lat, lon) coordinates.
Coordinates are min-max scaled to [0, 1] using the campus bounding box.

Labels returned per sample
--------------------------
    path_coords    FloatTensor (T, 2)   raw (before padding) — padded in collate
                                        columns: [lat_norm, lon_norm]
    path_len       int                  number of real waypoints
    belief_vector  FloatTensor (24,)    binary — use with BCEWithLogitsLoss
    desire_vector  FloatTensor (24,)    joint POI preference dist — use with KLDivLoss
    desire_idx     LongTensor  ()       0–5    — use with CrossEntropyLoss
    goal_poi_idx   LongTensor  ()       0–23   — use with CrossEntropyLoss

Metadata (strings, not tensors)
--------------------------
    agent_id, character, config, split

Usage
-----
from model.dataset import TrajectoryDataset, build_dataloaders

train_ds = TrajectoryDataset.from_jsonl("data/splits/train.jsonl", node_coords)
loader   = DataLoader(train_ds, batch_size=32, collate_fn=train_ds.collate_fn)

# Or build all four loaders at once:
loaders = build_dataloaders(
    splits_dir  = "data/splits",
    node_coords = node_coords,   # dict {node_id: (lat, lon)}
    batch_size  = 32,
)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from simulation.episode import EpisodeRecord


# ── Campus bounding box (used for coordinate normalisation) ───────────
_BBOX = {
    "lat_min": 32.867,  "lat_max": 32.895,
    "lon_min": -117.253, "lon_max": -117.220,
}

# ── Canonical orderings (must match poi_categories.json) ─────────────
CATEGORIES = [
    "attend_class", "get_food", "study",
    "socialize", "exercise", "rest_home",
]
CATEGORY_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

# Populated once by build_poi_index() — stable across all dataset instances
_POI_ORDER:  list[str]      = []   # poi_id strings, length 24
_POI_TO_IDX: dict[str, int] = {}


def build_poi_index(poi_cfg: dict) -> None:
    """
    Populate the module-level POI ordering from poi_categories.json.
    Must be called once before constructing any TrajectoryDataset.
    Called automatically via from_jsonl / from_records.
    """
    global _POI_ORDER, _POI_TO_IDX
    _POI_ORDER = [
        poi["id"]
        for cat_data in poi_cfg["categories"].values()
        for poi in cat_data["pois"]
    ]
    _POI_TO_IDX = {pid: i for i, pid in enumerate(_POI_ORDER)}


def node_coords_from_graph(G) -> dict[int, tuple[float, float]]:
    """
    Extract {node_id: (lat, lon)} from an osmnx graph.
    Call once after loading the graph and pass the result to the dataset.

    Example
    -------
    import osmnx as ox
    G = ox.load_graphml("data/ucsd_walk.graphml")
    node_coords = node_coords_from_graph(G)
    """
    return {
        int(node): (float(data["y"]), float(data["x"]))   # osmnx: y=lat, x=lon
        for node, data in G.nodes(data=True)
    }


def _normalise_coord(lat: float, lon: float) -> tuple[float, float]:
    """Min-max normalise (lat, lon) to [0, 1] using the campus bbox."""
    lat_n = (lat - _BBOX["lat_min"]) / (_BBOX["lat_max"] - _BBOX["lat_min"])
    lon_n = (lon - _BBOX["lon_min"]) / (_BBOX["lon_max"] - _BBOX["lon_min"])
    return lat_n, lon_n


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset over BDI ToM EpisodeRecords.

    Parameters
    ----------
    records     : list of EpisodeRecord (loaded from JSONL or produced by
                  SimulationRunner)
    node_coords : dict mapping OSM node_id → (lat, lon). Build with
                  node_coords_from_graph(G) after loading the osmnx graph.
    poi_cfg     : parsed poi_categories.json dict.  Needed only on first
                  construction to build the POI index; subsequent datasets
                  reuse the cached index.
    """

    def __init__(
        self,
        records: list[EpisodeRecord],
        node_coords: dict[int, tuple[float, float]],
        poi_cfg: dict | None = None,
    ):
        if poi_cfg is not None:
            build_poi_index(poi_cfg)
        if not _POI_ORDER:
            raise RuntimeError(
                "POI index not initialised. Pass poi_cfg= on first dataset construction."
            )

        self.records     = records
        self.node_coords = node_coords

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        node_coords: dict[int, tuple[float, float]],
        poi_cfg: dict | None = None,
    ) -> "TrajectoryDataset":
        """Load directly from a JSONL split file."""
        records: list[EpisodeRecord] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EpisodeRecord.from_json(line))
        return cls(records, node_coords, poi_cfg=poi_cfg)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]

        # ── Path coordinates ─────────────────────────────────────────
        coords = []
        for node_id in r.path:
            lat, lon = self.node_coords.get(node_id, (0.0, 0.0))
            lat_n, lon_n = _normalise_coord(lat, lon)
            coords.append((lat_n, lon_n))
        path_coords = torch.tensor(coords, dtype=torch.float32)  # (T, 2)

        # ── Labels ───────────────────────────────────────────────────
        belief_vector = torch.tensor(r.belief_vector,  dtype=torch.float32)  # (24,)
        desire_vector = torch.tensor(r.desire_vector,  dtype=torch.float32)  # (24,)
        desire_idx    = torch.tensor(CATEGORY_TO_IDX[r.desire], dtype=torch.long)
        goal_poi_idx  = torch.tensor(_POI_TO_IDX[r.goal_poi],   dtype=torch.long)

        return {
            # Variable-length — padded to batch max in collate_fn
            "path_coords":    path_coords,
            "path_len":       len(r.path),
            # Targets
            "belief_vector":  belief_vector,
            "desire_vector":  desire_vector,
            "desire_idx":     desire_idx,
            "goal_poi_idx":   goal_poi_idx,
            # Metadata (strings — not batched as tensors)
            "agent_id":       r.agent_id,
            "character":      r.character,
            "config":         r.belief_config,
            "split":          r.split,
        }

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Pad variable-length path_coords to the longest sequence in the batch
        and build an attention mask.

        Attention mask convention:
            1  →  real token
            0  →  padding

        Output shapes (B = batch size, T = max path length in batch):
            path_coords      FloatTensor  (B, T, 2)   [lat, lon]
            attention_mask   LongTensor   (B, T)
            path_lengths     LongTensor   (B,)
            belief_vector    FloatTensor  (B, 24)
            desire_vector    FloatTensor  (B, 24)
            desire_idx       LongTensor   (B,)
            goal_poi_idx     LongTensor   (B,)
            agent_id         list[str]    length B
            character        list[str]    length B
            config           list[str]    length B
            split            list[str]    length B
        """
        max_len = max(item["path_len"] for item in batch)
        B = len(batch)

        path_coords   = torch.zeros(B, max_len, 2, dtype=torch.float32)
        attn_mask     = torch.zeros(B, max_len, dtype=torch.long)

        belief_vecs   = torch.stack([item["belief_vector"]  for item in batch])
        desire_vecs   = torch.stack([item["desire_vector"]  for item in batch])
        desire_idxs   = torch.stack([item["desire_idx"]     for item in batch])
        goal_poi_idxs = torch.stack([item["goal_poi_idx"]   for item in batch])
        path_lengths  = torch.tensor([item["path_len"]     for item in batch], dtype=torch.long)

        for i, item in enumerate(batch):
            T = item["path_len"]
            path_coords[i, :T] = item["path_coords"]
            attn_mask[i,   :T] = 1

        return {
            "path_coords":    path_coords,
            "attention_mask": attn_mask,
            "path_lengths":   path_lengths,
            "belief_vector":  belief_vecs,
            "desire_vector":  desire_vecs,
            "desire_idx":     desire_idxs,
            "goal_poi_idx":   goal_poi_idxs,
            # Metadata
            "agent_id":   [item["agent_id"]  for item in batch],
            "character":  [item["character"] for item in batch],
            "config":     [item["config"]    for item in batch],
            "split":      [item["split"]     for item in batch],
        }


# ── Convenience builder ───────────────────────────────────────────────

def build_dataloaders(
    splits_dir: str | Path,
    node_coords: dict[int, tuple[float, float]],
    poi_cfg: dict,
    batch_size: int = 32,
    num_workers: int = 0,
    splits: list[str] | None = None,
) -> dict[str, DataLoader]:
    """
    Build a DataLoader for each available split file in splits_dir.

    Parameters
    ----------
    splits_dir  : directory containing {split}.jsonl files
    node_coords : {node_id: (lat, lon)} — build with node_coords_from_graph()
    poi_cfg     : parsed poi_categories.json
    batch_size  : samples per batch
    num_workers : DataLoader worker processes (0 = main process only)
    splits      : which splits to load; defaults to all .jsonl files found

    Returns
    -------
    dict mapping split name → DataLoader
    """
    splits_dir = Path(splits_dir)
    if splits is None:
        splits = [p.stem for p in sorted(splits_dir.glob("*.jsonl"))]

    loaders: dict[str, DataLoader] = {}
    first = True

    for split_name in splits:
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue

        ds = TrajectoryDataset.from_jsonl(
            path,
            node_coords,
            poi_cfg=poi_cfg if first else None,  # build index once
        )
        first = False

        shuffle = (split_name == "train")
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=TrajectoryDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  {split_name:<20} {len(ds):>6} samples  →  {len(loaders[split_name]):>4} batches (bs={batch_size})")

    return loaders
