"""
model/meta_dataset.py

MetaEpisodeDataset: constructs meta-learning episodes for BDIPredictor.

Each item pairs a current episode with two agent-specific context sets:

  desire_context   K trajectories sampled uniformly at random from the same
                   agent's other episodes.  Random sampling averages out
                   belief-induced filtering so the desire signal dominates.

  recent_context   The K_recent most-recent trajectories preceding the current
                   episode (same agent).  Recency ensures the window likely
                   reflects the current belief configuration.  The Belief head
                   sees both z_d and z_recent, letting it learn to detect which
                   POIs the agent normally visits (from z_d) but has been
                   avoiding lately (from z_recent) — the signal for false beliefs.

Both context sets are always drawn from the same agent as the current episode.

Usage
-----
from model.meta_dataset import MetaEpisodeDataset, build_meta_dataloaders

ds = MetaEpisodeDataset(records, node_coords, poi_to_idx, K=10, K_recent=5)
loader = DataLoader(ds, batch_size=32, collate_fn=MetaEpisodeDataset.collate_fn)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from simulation.episode import EpisodeRecord

# ── Coordinate normalisation (matches model/dataset.py) ───────────────
_BBOX = {
    "lat_min": 32.867,  "lat_max": 32.895,
    "lon_min": -117.253, "lon_max": -117.220,
}

def _norm(lat: float, lon: float) -> tuple[float, float]:
    lat_n = (lat - _BBOX["lat_min"]) / (_BBOX["lat_max"] - _BBOX["lat_min"])
    lon_n = (lon - _BBOX["lon_min"]) / (_BBOX["lon_max"] - _BBOX["lon_min"])
    return lat_n, lon_n


class MetaEpisodeDataset(Dataset):
    """
    Parameters
    ----------
    records     : list of EpisodeRecord (from a single split)
    node_coords : {node_id: (lat, lon)}  — build with node_coords_from_graph()
    poi_to_idx  : {poi_id: int}  — canonical POI index mapping
    K           : context size for both desire and belief nets
    seed        : RNG seed for reproducible desire sampling
    """

    def __init__(
        self,
        records:     list[EpisodeRecord],
        node_coords: dict[int, tuple[float, float]],
        poi_to_idx:  dict[str, int],
        K:           int = 10,
        K_recent:    int = 5,
        seed:        int = 0,
    ):
        self.records     = records
        self.node_coords = node_coords
        self.poi_to_idx  = poi_to_idx
        self.K           = K
        self.K_recent    = K_recent
        self.rng         = np.random.default_rng(seed)

        # Group episode flat-indices by agent, preserving insertion order
        # (which equals simulation order: config × episode_idx)
        self.agent_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, r in enumerate(records):
            self.agent_to_indices[r.agent_id].append(idx)

    # ── Dataset protocol ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r          = self.records[idx]
        agent_idxs = self.agent_to_indices[r.agent_id]   # ordered list

        # Position of current episode in this agent's ordered list
        pos    = agent_idxs.index(idx)
        others = [i for i in agent_idxs if i != idx]

        # ── Desire context: K random from same agent ──────────────
        if len(others) == 0:
            desire_src = [idx] * self.K
        elif len(others) < self.K:
            desire_src = self.rng.choice(others, size=self.K, replace=True).tolist()
        else:
            desire_src = self.rng.choice(others, size=self.K, replace=False).tolist()

        # ── Recent context: K_recent most-recent preceding episodes ──
        preceding = agent_idxs[:pos]
        if len(preceding) == 0:
            # No history yet — fall back to future episodes of same agent
            fallback = agent_idxs[pos + 1 :]
            if len(fallback) == 0:
                fallback = [idx]
            recent_src = _repeat_to(fallback, self.K_recent)
        elif len(preceding) < self.K_recent:
            recent_src = _repeat_to(preceding, self.K_recent)
        else:
            recent_src = preceding[-self.K_recent :]

        # ── Build tensors ─────────────────────────────────────────
        desire_coords, desire_mask = self._build_context(desire_src)
        recent_coords, recent_mask = self._build_context(recent_src)
        current_coords = self._record_to_coords(r)

        return {
            "desire_coords":  desire_coords,   # (K,        T_d, 2)  [lat, lon]
            "desire_mask":    desire_mask,      # (K,        T_d)
            "recent_coords":  recent_coords,    # (K_recent, T_r, 2)  [lat, lon]
            "recent_mask":    recent_mask,      # (K_recent, T_r)
            "current_coords": current_coords,   # (T_c, 2)  [lat, lon]
            "current_len":    len(r.path),
            "belief_vector":  torch.tensor(r.belief_vector, dtype=torch.float32),
            "desire_vector":  torch.tensor(r.desire_vector, dtype=torch.float32),
            "goal_poi_idx":   torch.tensor(self.poi_to_idx[r.goal_poi], dtype=torch.long),
            "agent_id":       r.agent_id,
            "character":      r.character,
            "config":         r.belief_config,
            "split":          r.split,
        }

    # ── Helpers ───────────────────────────────────────────────────────

    def _record_to_coords(self, r: EpisodeRecord) -> torch.Tensor:
        """Convert a record's path to a normalised (T, 2) float tensor [lat, lon]."""
        pts = [
            _norm(*self.node_coords.get(nid, (0.0, 0.0)))
            for nid in r.path
        ]
        return torch.tensor(pts, dtype=torch.float32)

    def _build_context(
        self, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build padded context tensors from a list of record indices.

        Returns
        -------
        coords : (K, T_max, 2)
        mask   : (K, T_max)  LongTensor, 1=real, 0=pad
        """
        tensors = [self._record_to_coords(self.records[i]) for i in indices]
        K     = len(tensors)
        lengths = [int(t.shape[0]) for t in tensors]
        T_max = max(lengths)
        coords = torch.zeros(K, T_max, 2)
        mask   = torch.zeros(K, T_max, dtype=torch.long)
        for i, (t, T) in enumerate(zip(tensors, lengths)):
            coords[i, :T] = t
            mask[i,   :T] = 1
        return coords, mask

    # ── Collation ─────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Pad variable-length context and current trajectories across the batch.

        Output shapes (B = batch size, K = desire context, K_r = recent context):
          desire_coords   FloatTensor (B, K,   T_d, 2)
          desire_mask     LongTensor  (B, K,   T_d)
          recent_coords   FloatTensor (B, K_r, T_r, 2)
          recent_mask     LongTensor  (B, K_r, T_r)
          current_coords  FloatTensor (B, T_c, 2)
          current_mask    LongTensor  (B, T_c)
          current_lengths LongTensor  (B,)
          belief_vector   FloatTensor (B, 24)
          desire_vector   FloatTensor (B, 24)
          goal_poi_idx    LongTensor  (B,)
        """
        B = len(batch)

        def pad_context(key_c: str, key_m: str):
            K_  = batch[0][key_c].size(0)
            T_max = max(item[key_c].size(1) for item in batch)
            coords = torch.zeros(B, K_, T_max, 2)
            mask   = torch.zeros(B, K_, T_max, dtype=torch.long)
            for i, item in enumerate(batch):
                T = item[key_c].size(1)
                coords[i, :, :T, :] = item[key_c]
                mask[i,   :, :T]    = item[key_m]
            return coords, mask

        desire_coords, desire_mask = pad_context("desire_coords", "desire_mask")
        recent_coords, recent_mask = pad_context("recent_coords", "recent_mask")

        T_c_max = max(item["current_len"] for item in batch)
        current_coords = torch.zeros(B, T_c_max, 2)
        current_mask   = torch.zeros(B, T_c_max, dtype=torch.long)
        for i, item in enumerate(batch):
            T = item["current_len"]
            current_coords[i, :T] = item["current_coords"]
            current_mask[i,   :T] = 1

        return {
            "desire_coords":  desire_coords,
            "desire_mask":    desire_mask,
            "recent_coords":  recent_coords,
            "recent_mask":    recent_mask,
            "current_coords": current_coords,
            "current_mask":   current_mask,
            "current_lengths": torch.tensor(
                [item["current_len"] for item in batch], dtype=torch.long
            ),
            "belief_vector": torch.stack([item["belief_vector"] for item in batch]),
            "desire_vector": torch.stack([item["desire_vector"] for item in batch]),
            "goal_poi_idx":  torch.stack([item["goal_poi_idx"]  for item in batch]),
            "agent_id":   [item["agent_id"]  for item in batch],
            "character":  [item["character"] for item in batch],
            "config":     [item["config"]    for item in batch],
            "split":      [item["split"]     for item in batch],
        }


# ── Helpers ────────────────────────────────────────────────────────────

def _repeat_to(lst: list, n: int) -> list:
    """Repeat lst cyclically until it has exactly n elements."""
    return (lst * ((n // len(lst)) + 1))[:n]


# ── Convenience builder ────────────────────────────────────────────────

def build_meta_dataloaders(
    splits_dir,
    node_coords:  dict[int, tuple[float, float]],
    poi_to_idx:   dict[str, int],
    K:            int  = 10,
    K_recent:     int  = 5,
    batch_size:   int  = 32,
    num_workers:  int  = 0,
    splits:       list[str] | None = None,
) -> dict[str, DataLoader]:
    """
    Build a MetaEpisodeDataset DataLoader for each available split file.

    Parameters
    ----------
    splits_dir  : directory containing {split}.jsonl files
    node_coords : {node_id: (lat, lon)}
    poi_to_idx  : {poi_id: int}
    K           : context set size
    batch_size  : samples per batch
    num_workers : DataLoader workers
    splits      : which splits to load; defaults to all .jsonl files found
    """
    from pathlib import Path
    splits_dir = Path(splits_dir)

    if splits is None:
        splits = [p.stem for p in sorted(splits_dir.glob("*.jsonl"))]

    loaders: dict[str, DataLoader] = {}
    for split_name in splits:
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue

        records: list[EpisodeRecord] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EpisodeRecord.from_json(line))

        ds = MetaEpisodeDataset(records, node_coords, poi_to_idx, K=K, K_recent=K_recent)
        shuffle = (split_name == "train")
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=MetaEpisodeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  {split_name:<20} {len(ds):>6} samples  "
              f"→  {len(loaders[split_name]):>4} batches (bs={batch_size}, K={K}, K_recent={K_recent})")

    return loaders
