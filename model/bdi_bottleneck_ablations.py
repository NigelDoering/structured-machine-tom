"""
model/bdi_bottleneck_ablations.py

Ablation variants of BDIBottleneckPredictor that remove one mental-state channel
to isolate each component's contribution to goal inference.

BDIBottleneckNoDesire
    Removes the desire encoder and desire head.  The model infers goal from
    belief state alone (derived from recent context) plus current trajectory.
    belief_head : Linear(d_model, num_pois)             — only z_r, no z_d
    goal_head input: cat[belief_probs (M), z_tau (d_model)]
    forward returns: (goal_logits, belief_logits)

BDIBottleneckNoBelief
    Removes the recent encoder and belief head.  The model infers goal from
    desire distribution alone (derived from desire context) plus current trajectory.
    desire_head : Linear(d_model, num_pois)             — only z_d
    goal_head input: cat[desire_probs (M), z_tau (d_model)]
    forward returns: (goal_logits, desire_logits)

Both retain the BDI bottleneck structure (goal head reads predicted probability
outputs rather than raw embeddings), so any accuracy difference vs the full
BDIBottleneckPredictor isolates the removed component's marginal contribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.goal_predictor import SinusoidalPE
from model.bdi_predictor import _make_encoder


class BDIBottleneckNoDesire(nn.Module):
    """
    BDIBottleneck without desire encoder/head.
    Goal is inferred from belief state (recent context) and current trajectory.
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

        self.recent_enc  = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.goal_enc    = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.recent_norm = nn.LayerNorm(d_model)
        self.goal_norm   = nn.LayerNorm(d_model)

        # belief_head reads only z_r (no z_d to concatenate)
        self.belief_head = nn.Linear(d_model, num_pois)

        # goal_head reads cat[belief_probs (M), z_tau (d_model)]
        self.goal_head = nn.Sequential(
            nn.Linear(num_pois + d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_pois),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.coord_proj, self.belief_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.goal_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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

    def forward(
        self,
        recent_coords:  torch.Tensor,   # (B, K_r, T_r, 2)
        recent_mask:    torch.Tensor,   # (B, K_r, T_r)
        current_coords: torch.Tensor,   # (B, T_c, 2)
        current_mask:   torch.Tensor,   # (B, T_c)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        goal_logits   : (B, M)  CrossEntropyLoss
        belief_logits : (B, M)  BCEWithLogitsLoss
        """
        z_r = self._encode_context(
            self.recent_enc, self.recent_norm, recent_coords, recent_mask
        )

        x_c   = self._embed(current_coords)
        x_c   = self.goal_enc(x_c, src_key_padding_mask=(current_mask == 0))
        x_c   = self.goal_norm(x_c)
        z_tau = self._mean_pool(x_c, current_mask)

        belief_logits = self.belief_head(z_r)
        belief_probs  = torch.sigmoid(belief_logits)

        goal_logits = self.goal_head(torch.cat([belief_probs, z_tau], dim=-1))
        return goal_logits, belief_logits


class BDIBottleneckNoBelief(nn.Module):
    """
    BDIBottleneck without recent encoder/belief head.
    Goal is inferred from desire distribution (desire context) and current trajectory.
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

        self.desire_enc  = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.goal_enc    = _make_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.desire_norm = nn.LayerNorm(d_model)
        self.goal_norm   = nn.LayerNorm(d_model)

        self.desire_head = nn.Linear(d_model, num_pois)

        # goal_head reads cat[desire_probs (M), z_tau (d_model)]
        self.goal_head = nn.Sequential(
            nn.Linear(num_pois + d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_pois),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.coord_proj, self.desire_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.goal_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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

    def forward(
        self,
        desire_coords:  torch.Tensor,   # (B, K, T_d, 2)
        desire_mask:    torch.Tensor,   # (B, K, T_d)
        current_coords: torch.Tensor,   # (B, T_c, 2)
        current_mask:   torch.Tensor,   # (B, T_c)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        goal_logits   : (B, M)  CrossEntropyLoss
        desire_logits : (B, M)  KLDivLoss (apply log_softmax before passing)
        """
        z_d = self._encode_context(
            self.desire_enc, self.desire_norm, desire_coords, desire_mask
        )

        x_c   = self._embed(current_coords)
        x_c   = self.goal_enc(x_c, src_key_padding_mask=(current_mask == 0))
        x_c   = self.goal_norm(x_c)
        z_tau = self._mean_pool(x_c, current_mask)

        desire_logits = self.desire_head(z_d)
        desire_probs  = F.softmax(desire_logits, dim=-1)

        goal_logits = self.goal_head(torch.cat([desire_probs, z_tau], dim=-1))
        return goal_logits, desire_logits
