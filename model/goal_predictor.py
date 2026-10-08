"""
model/goal_predictor.py

Simple transformer encoder that maps a (partial) pedestrian trajectory to a
goal POI distribution over 24 campus locations.

Architecture
------------
  coord_proj   Linear(2 → d_model)                — embed (lat, lon) coordinates
  cat_emb      Embedding(7, d_model)               — embed POI category tag (0=traversal)
  pos_enc      sinusoidal positional encoding      — sequence position
  encoder      TransformerEncoder (N layers)       — contextualise waypoints
  pool         masked mean-pool over real tokens   — aggregate to fixed vector
  head         Linear(d_model → num_pois)          — classify destination

Input
-----
  path_coords    FloatTensor (B, T, 3)   [lat_norm, lon_norm, cat_id], padded
  attention_mask LongTensor  (B, T)     1 = real token, 0 = padding

Output
------
  logits         FloatTensor (B, num_pois)   raw scores; apply softmax for probs
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class GoalPredictor(nn.Module):
    """
    Transformer goal predictor.

    Parameters
    ----------
    num_pois   : number of destination POI classes (default 24)
    d_model    : hidden dimension
    nhead      : number of attention heads
    num_layers : encoder depth
    dim_ff     : feedforward inner dimension
    dropout    : dropout probability
    """

    def __init__(
        self,
        num_pois:   int   = 24,
        d_model:    int   = 64,
        nhead:      int   = 4,
        num_layers: int   = 3,
        dim_ff:     int   = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.coord_proj = nn.Linear(2, d_model)
        self.pos_enc    = SinusoidalPE(d_model)
        self.drop       = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model          = d_model,
            nhead            = nhead,
            dim_feedforward  = dim_ff,
            dropout          = dropout,
            batch_first      = True,
            norm_first       = True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, num_pois)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.coord_proj.weight)
        nn.init.zeros_(self.coord_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        path_coords:    torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        path_coords    : FloatTensor (B, T, 3)  [lat_norm, lon_norm, cat_id] waypoints
        attention_mask : LongTensor  (B, T)     1 = real token, 0 = padding

        Returns
        -------
        logits : FloatTensor (B, num_pois)
        """
        node_emb = self.coord_proj(path_coords[..., :2])
        x   = self.drop(self.pos_enc(node_emb))                      # (B, T, d)
        kpm = attention_mask == 0                                     # True → ignore
        x   = self.encoder(x, src_key_padding_mask=kpm)              # (B, T, d)
        x   = self.norm(x)

        # Masked mean-pool — average only over real (non-padding) tokens
        mask_f = attention_mask.unsqueeze(-1).float()                 # (B, T, 1)
        pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9) # (B, d)

        return self.head(pooled)                                      # (B, num_pois)
