"""
model/bdi_predictor.py

BDIPredictor: three-stream transformer model for joint mental state attribution.

Streams
-------
  Desire Net  : K randomly-sampled past trajectories (same agent)
                → desire distribution  p̂_i ∈ Δ^{M-1}
  Recent Net  : K_recent most-recent past trajectories (same agent)
                → fed jointly with z_d into the Belief head
  Belief head : cat[z_d, z_recent] → soft belief vector b̂_i ∈ [0,1]^M
  Goal Net    : current partial trajectory + z_d + z_recent
                → goal logits over M POIs

The Belief head is comparative: it sees both the stable desire embedding z_d
("what this agent normally does") and the recent trajectory embedding z_recent
("what this agent has been doing lately").  The gap between them is the signal
for false beliefs — a POI the agent normally visits but has been avoiding
recently is likely believed closed.

Input shapes (B = batch, K = desire context, K_r = recent context,
              T = sequence length, M = num_pois)
------------------------------------------------------------------------------
  desire_coords   FloatTensor (B, K,   T_d, 3)   [lat, lon, cat_id]
  desire_mask     LongTensor  (B, K,   T_d)      1=real, 0=pad
  recent_coords   FloatTensor (B, K_r, T_r, 3)
  recent_mask     LongTensor  (B, K_r, T_r)
  current_coords  FloatTensor (B, T_c, 3)
  current_mask    LongTensor  (B, T_c)

Returns
-------
  goal_logits   FloatTensor (B, M)  — use with CrossEntropyLoss
  desire_logits FloatTensor (B, M)  — apply log_softmax, use with KLDivLoss
  belief_logits FloatTensor (B, M)  — use with BCEWithLogitsLoss
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.goal_predictor import SinusoidalPE


def _make_encoder(
    d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        dropout=dropout,
        batch_first=True,
        norm_first=True,   # pre-LN: more stable training
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class BDIPredictor(nn.Module):
    """
    Parameters
    ----------
    num_pois   : number of POI classes (M = 24)
    d_model    : transformer hidden dimension
    nhead      : attention heads per layer
    num_layers : encoder depth (same for all three encoders)
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

        # ── Shared input layers (all three streams) ───────────────────
        self.coord_proj = nn.Linear(2, d_model)
        self.pos_enc    = SinusoidalPE(d_model)
        self.drop       = nn.Dropout(dropout)

        # ── Three independent transformer encoders ────────────────────
        self.desire_enc  = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.recent_enc  = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.goal_enc    = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)

        self.desire_norm = nn.LayerNorm(d_model)
        self.recent_norm = nn.LayerNorm(d_model)
        self.goal_norm   = nn.LayerNorm(d_model)

        # ── Output heads ──────────────────────────────────────────────
        # Desire head  : z_d          (d)   → M
        # Belief head  : [z_d ‖ z_r]  (2*d) → M  (comparative: gap signals false beliefs)
        # Goal head    : [z_d ‖ z_r ‖ z_τ] (3*d) → M
        self.desire_head = nn.Linear(d_model,     num_pois)
        self.belief_head = nn.Linear(2 * d_model, num_pois)
        self.goal_head   = nn.Linear(3 * d_model, num_pois)

        self._init_weights()

    def _init_weights(self) -> None:
        for lin in (self.coord_proj, self.desire_head, self.belief_head, self.goal_head):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    # ── Internal helpers ──────────────────────────────────────────────

    def _embed(self, coords: torch.Tensor) -> torch.Tensor:
        """Coordinate projection + sinusoidal positional encoding."""
        node_emb = self.coord_proj(coords[..., :2])
        return self.drop(self.pos_enc(node_emb))

    def _mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean-pool over the sequence dimension.
        x    : (..., T, d)
        mask : (..., T)   1=real, 0=pad
        returns (..., d)
        """
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(-2) / mask_f.sum(-2).clamp(min=1e-9)

    def _encode_context(
        self,
        encoder: nn.TransformerEncoder,
        norm:    nn.LayerNorm,
        coords:  torch.Tensor,   # (B, K, T, 2)
        mask:    torch.Tensor,   # (B, K, T)
    ) -> torch.Tensor:           # (B, d)
        """
        Encode K context trajectories per sample and return their mean embedding.

        The K trajectories are flattened into a single batch of B*K sequences,
        processed in parallel, then averaged back to B embeddings.
        """
        B, K, T, _ = coords.shape
        x = self._embed(coords.view(B * K, T, 2))               # (B*K, T, d)
        m = mask.view(B * K, T)
        x = encoder(x, src_key_padding_mask=(m == 0))            # (B*K, T, d)
        x = norm(x)
        pooled = self._mean_pool(x, m)                           # (B*K, d)
        return pooled.view(B, K, -1).mean(dim=1)                 # (B, d)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        desire_coords:  torch.Tensor,   # (B, K,   T_d, 3)
        desire_mask:    torch.Tensor,   # (B, K,   T_d)
        recent_coords:  torch.Tensor,   # (B, K_r, T_r, 3)
        recent_mask:    torch.Tensor,   # (B, K_r, T_r)
        current_coords: torch.Tensor,   # (B, T_c, 3)
        current_mask:   torch.Tensor,   # (B, T_c)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        goal_logits   : (B, M)  CrossEntropyLoss
        desire_logits : (B, M)  KLDivLoss (apply log_softmax before passing)
        belief_logits : (B, M)  BCEWithLogitsLoss
        """
        # Stable preference signal: K random past trajectories
        z_d = self._encode_context(
            self.desire_enc, self.desire_norm, desire_coords, desire_mask
        )   # (B, d)

        # Recent behaviour signal: K_recent most-recent past trajectories
        z_r = self._encode_context(
            self.recent_enc, self.recent_norm, recent_coords, recent_mask
        )   # (B, d)

        # Current trajectory embedding
        x_c   = self._embed(current_coords)
        x_c   = self.goal_enc(x_c, src_key_padding_mask=(current_mask == 0))
        x_c   = self.goal_norm(x_c)
        z_tau = self._mean_pool(x_c, current_mask)                # (B, d)

        # Heads
        # Belief head sees both z_d and z_r: gap between stable preference and
        # recent behaviour is the evidence for false beliefs.
        belief_logits = self.belief_head(torch.cat([z_d, z_r], dim=-1))
        desire_logits = self.desire_head(z_d)
        goal_logits   = self.goal_head(torch.cat([z_d, z_r, z_tau], dim=-1))

        return goal_logits, desire_logits, belief_logits
