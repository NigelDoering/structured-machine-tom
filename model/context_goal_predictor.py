"""
model/context_goal_predictor.py

ContextGoalPredictor: agent-aware goal prediction baseline.

Motivation
----------
GoalPredictor (the primary baseline) has no notion of agent identity —
it sees only the current partial trajectory.  BDIBottleneckPredictor has
access to agent-stratified context through its desire and recent encoders,
which implicitly carry agent identity information.  A fair comparison needs
a baseline that also receives this context but without the structured
belief/desire bottleneck.

This model has the same three-stream transformer architecture as
BDIBottleneckPredictor but omits the belief and desire prediction heads.
The encoded context embeddings (z_d, z_r) are concatenated with the current
trajectory embedding (z_tau) and fed directly into a goal MLP.

Architecture
------------
  Inputs:
    desire_coords  (B, K,   T_d, 2)  — K random past episodes, same agent
    desire_mask    (B, K,   T_d)
    recent_coords  (B, K_r, T_r, 3)  — K_r most-recent past episodes
    recent_mask    (B, K_r, T_r)
    current_coords (B, T_c, 3)       — current partial trajectory
    current_mask   (B, T_c)

  Encoders (independent transformers, identical to BDIBottleneck):
    z_d   = mean_pool(desire_enc(desire_coords))       ∈ R^{d_model}
    z_r   = mean_pool(recent_enc(recent_coords))       ∈ R^{d_model}
    z_tau = mean_pool(goal_enc(current_coords))        ∈ R^{d_model}

  Goal head (2-layer MLP):
    goal_logits = MLP(cat[z_d, z_r, z_tau])           ∈ R^M
    input size  = 3 * d_model

  No belief head.  No desire head.  Single cross-entropy loss.

Comparison structure
--------------------
  GoalPredictor          : no agent context, single-stream
  ContextGoalPredictor   : agent context, three-stream, no structured heads
  BDIBottleneckPredictor : agent context, three-stream, explicit belief+desire heads

The gap between ContextGoalPredictor and BDIBottleneckPredictor isolates the
contribution of the structured mental-state bottleneck, independent of agent
identity access.

Returns
-------
  goal_logits : FloatTensor (B, M)  — CrossEntropyLoss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.goal_predictor import SinusoidalPE
from model.bdi_predictor import _make_encoder


class ContextGoalPredictor(nn.Module):
    """
    Parameters
    ----------
    num_pois   : number of POI classes (M = 24)
    d_model    : transformer hidden dimension
    nhead      : attention heads per layer
    num_layers : encoder depth
    dim_ff     : feedforward inner dimension (also goal MLP hidden size)
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

        # ── Shared input layers ───────────────────────────────────────
        self.coord_proj = nn.Linear(2, d_model)
        self.pos_enc    = SinusoidalPE(d_model)
        self.drop       = nn.Dropout(dropout)

        # ── Three independent transformer encoders ────────────────────
        # Same structure as BDIBottleneckPredictor so parameter counts
        # and representational capacity are matched.
        self.desire_enc  = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.recent_enc  = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.goal_enc    = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)

        self.desire_norm = nn.LayerNorm(d_model)
        self.recent_norm = nn.LayerNorm(d_model)
        self.goal_norm   = nn.LayerNorm(d_model)

        # ── Goal head: 2-layer MLP ────────────────────────────────────
        # Input: cat[z_d (d), z_r (d), z_tau (d)]  →  3 * d_model dims.
        # Same hidden size and nonlinearity as BDIBottleneck goal head
        # so the only difference is what sits before the MLP input.
        self.goal_head = nn.Sequential(
            nn.Linear(3 * d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_pois),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.coord_proj.weight)
        nn.init.zeros_(self.coord_proj.bias)
        for m in self.goal_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Internal helpers (identical to BDIBottleneckPredictor) ────────

    def _embed(self, coords: torch.Tensor) -> torch.Tensor:
        node_emb = self.coord_proj(coords[..., :2])
        return self.drop(self.pos_enc(node_emb))

    def _mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(-2) / mask_f.sum(-2).clamp(min=1e-9)

    def _encode_context(
        self,
        encoder: nn.TransformerEncoder,
        norm:    nn.LayerNorm,
        coords:  torch.Tensor,   # (B, K, T, 2)
        mask:    torch.Tensor,   # (B, K, T)
    ) -> torch.Tensor:           # (B, d_model)
        B, K, T, _ = coords.shape
        x = self._embed(coords.view(B * K, T, 2))
        m = mask.view(B * K, T)
        x = encoder(x, src_key_padding_mask=(m == 0))
        x = norm(x)
        pooled = self._mean_pool(x, m)
        return pooled.view(B, K, -1).mean(dim=1)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        desire_coords:  torch.Tensor,   # (B, K,   T_d, 2)
        desire_mask:    torch.Tensor,   # (B, K,   T_d)
        recent_coords:  torch.Tensor,   # (B, K_r, T_r, 3)
        recent_mask:    torch.Tensor,   # (B, K_r, T_r)
        current_coords: torch.Tensor,   # (B, T_c, 3)
        current_mask:   torch.Tensor,   # (B, T_c)
    ) -> torch.Tensor:                  # (B, M)
        """
        Returns
        -------
        goal_logits : (B, M)  — pass to F.cross_entropy
        """
        z_d = self._encode_context(
            self.desire_enc, self.desire_norm, desire_coords, desire_mask
        )
        z_r = self._encode_context(
            self.recent_enc, self.recent_norm, recent_coords, recent_mask
        )

        x_c   = self._embed(current_coords)
        x_c   = self.goal_enc(x_c, src_key_padding_mask=(current_mask == 0))
        x_c   = self.goal_norm(x_c)
        z_tau = self._mean_pool(x_c, current_mask)

        return self.goal_head(torch.cat([z_d, z_r, z_tau], dim=-1))


class AuxSupervisedContextGoalPredictor(ContextGoalPredictor):
    """
    ContextGoalPredictor with auxiliary desire and belief supervision.

    The goal head still consumes raw encoder embeddings cat[z_d, z_r, z_tau],
    not the predicted desire/belief probabilities. This isolates auxiliary
    mental-state supervision from the structured bottleneck pathway.
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
        super().__init__(
            num_pois=num_pois,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.desire_head = nn.Linear(d_model, num_pois)
        self.belief_head = nn.Linear(2 * d_model, num_pois)
        nn.init.xavier_uniform_(self.desire_head.weight)
        nn.init.zeros_(self.desire_head.bias)
        nn.init.xavier_uniform_(self.belief_head.weight)
        nn.init.zeros_(self.belief_head.bias)

    def encode_embeddings(
        self,
        desire_coords:  torch.Tensor,
        desire_mask:    torch.Tensor,
        recent_coords:  torch.Tensor,
        recent_mask:    torch.Tensor,
        current_coords: torch.Tensor,
        current_mask:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw z_d, z_r, z_tau embeddings used by the unbottlenecked goal head."""
        z_d = self._encode_context(
            self.desire_enc, self.desire_norm, desire_coords, desire_mask
        )
        z_r = self._encode_context(
            self.recent_enc, self.recent_norm, recent_coords, recent_mask
        )

        x_c = self._embed(current_coords)
        x_c = self.goal_enc(x_c, src_key_padding_mask=(current_mask == 0))
        x_c = self.goal_norm(x_c)
        z_tau = self._mean_pool(x_c, current_mask)
        return z_d, z_r, z_tau

    def forward(
        self,
        desire_coords:  torch.Tensor,
        desire_mask:    torch.Tensor,
        recent_coords:  torch.Tensor,
        recent_mask:    torch.Tensor,
        current_coords: torch.Tensor,
        current_mask:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_d, z_r, z_tau = self.encode_embeddings(
            desire_coords,
            desire_mask,
            recent_coords,
            recent_mask,
            current_coords,
            current_mask,
        )
        goal_logits = self.goal_head(torch.cat([z_d, z_r, z_tau], dim=-1))
        desire_logits = self.desire_head(z_d)
        belief_logits = self.belief_head(torch.cat([z_d, z_r], dim=-1))
        return goal_logits, desire_logits, belief_logits

    def predict_aux(
        self,
        desire_coords:  torch.Tensor,
        desire_mask:    torch.Tensor,
        recent_coords:  torch.Tensor,
        recent_mask:    torch.Tensor,
        current_coords: torch.Tensor,
        current_mask:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probability-space auxiliary predictions for diagnostics."""
        _, desire_logits, belief_logits = self(
            desire_coords,
            desire_mask,
            recent_coords,
            recent_mask,
            current_coords,
            current_mask,
        )
        return F.softmax(desire_logits, dim=-1), torch.sigmoid(belief_logits)
