"""
model/bdi_bottleneck.py

BDIBottleneckPredictor: architectural variant of BDIPredictor in which the
goal head receives the *outputs* of the belief and desire prediction heads
rather than the raw transformer embeddings.

Motivation
----------
In BDIPredictor the goal head reads cat[z_d, z_r, z_tau] — all raw embeddings.
This makes counterfactual interventions on beliefs or desires impossible because
there is no explicit slot to modify.

Here the goal head instead reads cat[desire_probs, belief_probs, z_tau], where
  desire_probs = softmax(desire_head(z_d))         ∈ Δ^{M-1}
  belief_probs = sigmoid(belief_head([z_d ‖ z_r])) ∈ [0,1]^M

At inference time a counterfactual is trivially: run the encoders once, obtain
desire_probs and belief_probs, modify any entry, re-run only the goal head.

Architecture differences vs. BDIPredictor
------------------------------------------
1. Goal head input: 2*M + d_model  (was 3*d_model)
2. Goal head is a 2-layer MLP with a ReLU hidden layer of size dim_ff.
   A linear layer on cat[desire_probs, belief_probs] cannot represent the
   element-wise desire×belief product that underlies the BDI decision rule;
   a nonlinear head can learn this interaction.
3. Everything else (encoders, belief head, desire head, losses) is identical.

Returns
-------
  goal_logits   FloatTensor (B, M)  — CrossEntropyLoss
  desire_logits FloatTensor (B, M)  — KLDivLoss  (apply log_softmax first)
  belief_logits FloatTensor (B, M)  — BCEWithLogitsLoss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.goal_predictor import SinusoidalPE
from model.bdi_predictor import _make_encoder


class BDIBottleneckPredictor(nn.Module):
    """
    Parameters
    ----------
    num_pois   : number of POI classes (M = 24)
    d_model    : transformer hidden dimension
    nhead      : attention heads per layer
    num_layers : encoder depth
    dim_ff     : feedforward inner dimension (also used as goal MLP hidden dim)
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
        self.desire_enc  = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.recent_enc  = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.goal_enc    = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)

        self.desire_norm = nn.LayerNorm(d_model)
        self.recent_norm = nn.LayerNorm(d_model)
        self.goal_norm   = nn.LayerNorm(d_model)

        # ── Prediction heads (identical to BDIPredictor) ──────────────
        self.desire_head = nn.Linear(d_model,     num_pois)
        self.belief_head = nn.Linear(2 * d_model, num_pois)

        # ── Goal head: 2-layer MLP ────────────────────────────────────
        # Input: cat[desire_probs (M), belief_probs (M), z_tau (d_model)]
        # Hidden: dim_ff with ReLU — can learn the desire × belief product
        self.goal_head = nn.Sequential(
            nn.Linear(2 * num_pois + d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_pois),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.coord_proj, self.desire_head, self.belief_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # MLP layers
        for m in self.goal_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Internal helpers (identical to BDIPredictor) ──────────────────

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
        coords:  torch.Tensor,
        mask:    torch.Tensor,
    ) -> torch.Tensor:
        B, K, T, _ = coords.shape
        x = self._embed(coords.view(B * K, T, 2))
        m = mask.view(B * K, T)
        x = encoder(x, src_key_padding_mask=(m == 0))
        x = norm(x)
        pooled = self._mean_pool(x, m)
        return pooled.view(B, K, -1).mean(dim=1)

    # ── Counterfactual helpers ────────────────────────────────────────

    def encode_components(
        self,
        desire_coords:  torch.Tensor,   # (B, K,   T_d, 2)
        desire_mask:    torch.Tensor,
        recent_coords:  torch.Tensor,   # (B, K_r, T_r, 3)
        recent_mask:    torch.Tensor,
        current_coords: torch.Tensor,   # (B, T_c, 3)
        current_mask:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run all encoders and prediction heads but stop before the goal head.

        Returns
        -------
        desire_probs : (B, M)       softmax(desire_logits)
        belief_probs : (B, M)       sigmoid(belief_logits)
        z_tau        : (B, d_model) current-trajectory embedding
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

        desire_logits = self.desire_head(z_d)
        belief_logits = self.belief_head(torch.cat([z_d, z_r], dim=-1))

        return (
            F.softmax(desire_logits, dim=-1),
            torch.sigmoid(belief_logits),
            z_tau,
        )

    def predict_goal(
        self,
        desire_probs: torch.Tensor,  # (B, M)
        belief_probs: torch.Tensor,  # (B, M)
        z_tau:        torch.Tensor,  # (B, d_model)
    ) -> torch.Tensor:
        """Run only the goal head. Accepts counterfactual desire/belief inputs."""
        return self.goal_head(
            torch.cat([desire_probs, belief_probs, z_tau], dim=-1)
        )

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        desire_coords:  torch.Tensor,   # (B, K,   T_d, 2)
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
        # Context embeddings
        z_d = self._encode_context(
            self.desire_enc, self.desire_norm, desire_coords, desire_mask
        )
        z_r = self._encode_context(
            self.recent_enc, self.recent_norm, recent_coords, recent_mask
        )

        # Current trajectory embedding
        x_c   = self._embed(current_coords)
        x_c   = self.goal_enc(x_c, src_key_padding_mask=(current_mask == 0))
        x_c   = self.goal_norm(x_c)
        z_tau = self._mean_pool(x_c, current_mask)

        # Prediction heads
        desire_logits = self.desire_head(z_d)
        belief_logits = self.belief_head(torch.cat([z_d, z_r], dim=-1))

        # Bottleneck: convert to probabilities and feed to goal head.
        # Gradients flow back through belief_logits and desire_logits so the
        # goal loss jointly trains the prediction heads end-to-end.
        desire_probs = F.softmax(desire_logits, dim=-1)          # (B, M)
        belief_probs = torch.sigmoid(belief_logits)               # (B, M)

        goal_logits = self.goal_head(
            torch.cat([desire_probs, belief_probs, z_tau], dim=-1)
        )

        return goal_logits, desire_logits, belief_logits
