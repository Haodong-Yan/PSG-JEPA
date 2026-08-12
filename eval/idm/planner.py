"""Pairwise IDM planning via latent interpolation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from idm.model import PairwiseIDM, GoalConditionedIDM, IDMConfig


@dataclass
class IDMPlanConfig:
    horizon: int = 10
    receding_horizon: int = 5
    refinement_steps: int = 0
    n_candidates: int = 1
    candidate_noise: float = 1.0


class IDMPlanner:
    """Lerp interpolation + pairwise IDM action decoding."""

    def __init__(self, jepa_model: nn.Module, idm: PairwiseIDM,
                 config: IDMPlanConfig, device: torch.device):
        self.jepa = jepa_model
        self.idm = idm
        self.config = config
        self.device = device
        self.jepa.eval()
        self.idm.eval()

    @torch.no_grad()
    def encode_obs(self, pixels: Tensor) -> Tensor:
        output = self.jepa.encoder(pixels, interpolate_pos_encoding=True)
        return self.jepa.projector(output.last_hidden_state[:, 0])

    @staticmethod
    def _lerp(z_start: Tensor, z_goal: Tensor, steps: int) -> Tensor:
        alphas = torch.linspace(0, 1, steps + 1, device=z_start.device)
        z_start = z_start.unsqueeze(1)
        z_goal = z_goal.unsqueeze(1)
        return z_start + alphas.view(1, -1, 1) * (z_goal - z_start)

    @torch.no_grad()
    def plan(self, z_start: Tensor, z_goal: Tensor) -> Tensor:
        """(B, D), (B, D) → (B, horizon, effective_act_dim)"""
        H = self.config.horizon
        z_traj = self._lerp(z_start, z_goal, H)
        actions = self._decode_trajectory(z_traj)
        return actions

    @torch.no_grad()
    def _decode_trajectory(self, z_traj: Tensor) -> Tensor:
        H = z_traj.shape[1] - 1
        actions = [self.idm(z_traj[:, t], z_traj[:, t + 1]) for t in range(H)]
        return torch.stack(actions, dim=1)


class IDMSolver:
    """Drop-in solver interface wrapping IDMPlanner."""

    def __init__(self, jepa_model: nn.Module, idm: PairwiseIDM,
                 config: IDMPlanConfig, device: torch.device):
        self.planner = IDMPlanner(jepa_model, idm, config, device)
        self.config = config
        self.device = device
        self._timings: list[float] = []

    @torch.no_grad()
    def __call__(self, info_dict: dict) -> Tensor:
        t0 = time.time()
        pixels = info_dict["pixels"]
        if pixels.ndim == 5:
            pixels = pixels[:, -1]
        pixels = pixels.float().to(self.device)
        z_start = self.planner.encode_obs(pixels)

        goal = info_dict["goal"]
        if goal.ndim == 5:
            goal = goal[:, -1]
        goal = goal.float().to(self.device)
        z_goal = self.planner.encode_obs(goal)

        actions = self.planner.plan(z_start, z_goal)
        self._timings.append(time.time() - t0)
        return actions
