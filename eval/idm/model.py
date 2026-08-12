"""Inverse Dynamics Models for LeWM latent space."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class IDMConfig:
    """Configuration for Inverse Dynamics Models."""

    embed_dim: int = 192          # LeWM embedding dimension
    action_dim: int = 2           # raw action dim
    frameskip: int = 1            # actions per step (effective_act_dim = action_dim * frameskip)
    hidden_dim: int = 512         # MLP hidden dimension
    n_layers: int = 3             # number of hidden layers
    dropout: float = 0.1
    noise_sigma: float = 0.0     # Gaussian noise on input embeddings during training
    noise_schedule: str = "fixed"  # "fixed" | "uniform"
    activation: str = "gelu"
    max_horizon: int = 50         # horizon normalizer for GoalConditionedIDM


class GoalConditionedIDM(nn.Module):
    """Goal-Conditioned IDM: (z_t, z_goal, steps_remaining) → a_t

    Single forward pass per step with AdaLN-Zero horizon modulation.
    """

    def __init__(self, cfg: IDMConfig):
        super().__init__()
        self.cfg = cfg
        self.effective_act_dim = cfg.action_dim * cfg.frameskip
        self.max_horizon = cfg.max_horizon

        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[cfg.activation]

        horizon_dim = 64
        self.horizon_embed = nn.Sequential(
            nn.Linear(horizon_dim, cfg.hidden_dim),
            act_fn(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        layers: list[nn.Module] = []
        in_dim = cfg.embed_dim * 2
        for _ in range(cfg.n_layers):
            layers.extend([
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                act_fn(),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = cfg.hidden_dim

        self.backbone = nn.Sequential(*layers)

        # AdaLN-Zero modulation
        self.ada_scale = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.ada_shift = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.head = nn.Linear(cfg.hidden_dim, self.effective_act_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        # AdaLN-Zero: start as identity (h * 1 + 0 = h)
        nn.init.zeros_(self.ada_scale.weight)
        nn.init.zeros_(self.ada_scale.bias)
        nn.init.zeros_(self.ada_shift.weight)
        nn.init.zeros_(self.ada_shift.bias)

    @staticmethod
    def _sinusoidal_embed(t: Tensor, dim: int = 64) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def _apply_noise(self, z: Tensor) -> Tensor:
        if not self.training or self.cfg.noise_sigma <= 0:
            return z
        return z + self.cfg.noise_sigma * torch.randn_like(z)

    def forward(self, z_t: Tensor, z_goal: Tensor, steps_remaining: Tensor) -> Tensor:
        """(B, D), (B, D), (B,) → (B, effective_act_dim)"""
        z_t = self._apply_noise(z_t)
        z_goal = self._apply_noise(z_goal)

        h_frac = steps_remaining.float() / self.max_horizon
        h_emb = self.horizon_embed(self._sinusoidal_embed(h_frac))

        h = torch.cat([z_t, z_goal], dim=-1)
        h = self.backbone(h)

        scale = self.ada_scale(h_emb)
        shift = self.ada_shift(h_emb)
        h = h * (1.0 + scale) + shift

        return self.head(h)


