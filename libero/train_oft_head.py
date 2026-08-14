#!/usr/bin/env python3
"""Train an OFT-style LIBERO-Goal action chunk head on a frozen LeWM encoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import tv_tensors


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "libero"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import (  # noqa: E402
    DEFAULT_LEWM,
    LIBERO_ROOT,
    LEWM_CODE,
    demo_sort_key,
    load_world_model,
    make_image_transform,
    parse_image_keys,
    parse_task_ids,
    prepend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument("--init-policy", type=Path, default=DEFAULT_LEWM,
                        help="pretrained encoder checkpoint; required unless --encoder-kind dinov2_hf")
    parser.add_argument("--encoder-kind", choices=("lewm", "dinov2_hf"), default="lewm")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional saved LeWM+OFT checkpoint to initialize both world model and head from.",
    )
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=-1,
        help="Epoch number offset for resumed/fine-tuned checkpoints. -1 infers from --resume-checkpoint filename.",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "libero_goal_lewm_oft_head")
    parser.add_argument("--tasks", default="all")
    parser.add_argument(
        "--task-repeat",
        default="",
        help="Optional task oversampling factors, e.g. '2:2,5:3,6:3'. Factor is total copies per sample.",
    )
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--image-keys", default="agentview_rgb,eye_in_hand_rgb")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotate-images-180", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-noop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noop-threshold", type=float, default=1e-4)
    parser.add_argument("--normalization", choices=("bounds_q99", "zscore"), default="bounds_q99")
    parser.add_argument("--normalize-gripper-action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-type", choices=("oft", "groot_fm", "action_query", "act_cvae"), default="oft")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fm-sampling-steps", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--kl-beta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--loss", choices=("l1", "smooth_l1"), default="l1")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--train-encoder", action="store_true")
    return parser.parse_args()


def parse_task_repeat(value: str) -> dict[int, int]:
    repeats: dict[int, int] = {}
    if not value.strip():
        return repeats
    for part in value.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"Bad --task-repeat entry {part!r}; expected TASK:FACTOR")
        task, factor = part.split(":", 1)
        task_id = int(task.strip())
        repeat = int(factor.strip())
        if task_id < 0 or repeat < 1:
            raise ValueError(f"Bad --task-repeat entry {part!r}; task >= 0 and factor >= 1 required")
        repeats[task_id] = repeat
    return repeats


def infer_epoch_offset(path: Path | None) -> int:
    if path is None:
        return 0
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else 0


class ZScoreScaler:
    def __init__(self, values: np.ndarray, mask: np.ndarray | None = None):
        self.mean = values.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = np.maximum(values.std(axis=0, keepdims=True).astype(np.float32), 1e-6)
        self.mask = np.ones_like(self.mean, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).reshape(1, -1)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        normalized = (values - self.mean) / self.std
        return np.where(self.mask, normalized, values)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        restored = values * self.std + self.mean
        return np.where(self.mask, restored, values)

    def state_dict(self) -> dict[str, np.ndarray | str]:
        return {"kind": "zscore", "mean": self.mean, "std": self.std, "mask": self.mask}


class BoundsQ99Scaler:
    def __init__(self, values: np.ndarray, mask: np.ndarray | None = None):
        self.low = np.quantile(values, 0.01, axis=0, keepdims=True).astype(np.float32)
        self.high = np.quantile(values, 0.99, axis=0, keepdims=True).astype(np.float32)
        self.span = np.maximum(self.high - self.low, 1e-6).astype(np.float32)
        self.mask = np.ones_like(self.low, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).reshape(1, -1)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        normalized = np.clip(2.0 * (values - self.low) / self.span - 1.0, -1.0, 1.0)
        return np.where(self.mask, normalized, values)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        restored = (values + 1.0) * 0.5 * self.span + self.low
        return np.where(self.mask, restored, values)

    def state_dict(self) -> dict[str, np.ndarray | str]:
        return {"kind": "bounds_q99", "low": self.low, "high": self.high, "mask": self.mask}


def make_scaler(values: np.ndarray, kind: str, mask: np.ndarray | None = None) -> ZScoreScaler | BoundsQ99Scaler:
    if kind == "zscore":
        return ZScoreScaler(values, mask=mask)
    return BoundsQ99Scaler(values, mask=mask)


def preprocess_libero_action(actions: np.ndarray) -> np.ndarray:
    """Convert LIBERO env gripper convention to OpenVLA-OFT training convention."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    gripper = np.clip(actions[..., -1:], 0.0, 1.0)
    actions[..., -1:] = 1.0 - gripper  # 1=open, 0=close
    return actions


def demo_oft_proprio(demo: h5py.Group, indices: Any) -> np.ndarray:
    ee = np.asarray(demo["obs"]["ee_states"][indices], dtype=np.float32)
    gripper = np.asarray(demo["obs"]["gripper_states"][indices], dtype=np.float32)
    return np.concatenate([ee, gripper], axis=-1)


class LiberoGoalLeWMOFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        libero_root: Path,
        transform: Any,
        img_size: int,
        seq_len: int,
        chunk_len: int,
        image_keys: list[str],
        task_ids: list[int],
        center_crop: bool,
        rotate_images_180: bool,
        filter_noop: bool,
        noop_threshold: float,
        normalization: str,
        normalize_gripper_action: bool,
        task_repeat: dict[int, int] | None = None,
    ):
        self.libero_root = libero_root
        self.transform = transform
        self.img_size = img_size
        self.seq_len = seq_len
        self.chunk_len = chunk_len
        self.image_keys = image_keys
        self.task_ids = task_ids
        self.center_crop = center_crop
        self.rotate_images_180 = rotate_images_180
        self.files: list[Path] = []
        self.samples: list[tuple[int, int, str, int]] = []
        self._handles: dict[int, h5py.File] = {}
        self.task_repeat = task_repeat or {}

        prepend(libero_root)
        from libero.libero import benchmark  # noqa: WPS433

        bench = benchmark.get_benchmark_dict()["libero_goal"]()
        action_values, proprio_values = [], []
        for task_id in self.task_ids:
            path = libero_root / "datasets" / bench.get_task_demonstration(task_id)
            file_idx = len(self.files)
            self.files.append(path)
            with h5py.File(path, "r") as handle:
                demos = handle["data"]
                for demo_name in sorted(demos.keys(), key=demo_sort_key):
                    demo = demos[demo_name]
                    raw_actions = np.asarray(demo["actions"], dtype=np.float32)
                    actions = preprocess_libero_action(raw_actions)
                    length = int(actions.shape[0])
                    first = max(0, self.seq_len - 1)
                    last = length - self.chunk_len
                    for step in range(first, last + 1):
                        if filter_noop and np.max(np.abs(raw_actions[step, :6])) < noop_threshold:
                            continue
                        self.samples.append((task_id, file_idx, demo_name, step))
                    action_values.append(actions)
                    proprio_values.append(demo_oft_proprio(demo, slice(None)))

        if not self.samples:
            raise RuntimeError("No LIBERO training samples survived filtering")
        if self.task_repeat:
            original_samples = self.samples
            repeated: list[tuple[int, int, str, int]] = []
            for sample in original_samples:
                repeat = self.task_repeat.get(sample[0], 1)
                repeated.extend([sample] * repeat)
            self.samples = repeated
        action_arr = np.concatenate(action_values, axis=0)
        proprio_arr = np.concatenate(proprio_values, axis=0)
        action_mask = np.ones((action_arr.shape[-1],), dtype=bool)
        if not normalize_gripper_action:
            action_mask[-1] = False
        self.action_scaler = make_scaler(action_arr, normalization, mask=action_mask)
        self.proprio_scaler = make_scaler(proprio_arr, normalization)

    def __len__(self) -> int:
        return len(self.samples)

    def _handle(self, file_idx: int) -> h5py.File:
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self.files[file_idx], "r")
        return self._handles[file_idx]

    def _prep_image(self, image: np.ndarray) -> torch.Tensor:
        if self.rotate_images_180:
            image = image[::-1, ::-1]
        if self.center_crop:
            height, width = image.shape[:2]
            crop_h, crop_w = int(height * 0.9), int(width * 0.9)
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            image = image[top : top + crop_h, left : left + crop_w]
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize((self.img_size, self.img_size), Image.BILINEAR))
        image = np.transpose(image.astype(np.uint8, copy=False), (2, 0, 1))
        return self.transform(tv_tensors.Image(image))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task_id, file_idx, demo_name, step = self.samples[index]
        demo = self._handle(file_idx)["data"][demo_name]
        obs_start = step - self.seq_len + 1
        obs_indices = list(range(obs_start, step + 1))
        action_indices = list(range(step, step + self.chunk_len))
        pixels = torch.stack(
            [
                torch.stack([self._prep_image(np.asarray(demo["obs"][key][i])) for key in self.image_keys], dim=0)
                for i in obs_indices
            ],
            dim=0,
        )
        actions = self.action_scaler.transform(preprocess_libero_action(np.asarray(demo["actions"][action_indices], dtype=np.float32)))
        proprios = self.proprio_scaler.transform(demo_oft_proprio(demo, obs_indices))
        task = np.zeros((10,), dtype=np.float32)
        task[task_id] = 1.0
        task_seq = np.repeat(task[None, :], self.seq_len, axis=0)
        return {
            "pixels": pixels,
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "proprios": torch.from_numpy(proprios.astype(np.float32)),
            "tasks": torch.from_numpy(task_seq.astype(np.float32)),
        }


class LeWMLiberoOFTHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        proprio_dim: int,
        task_dim: int,
        action_dim: int,
        seq_len: int,
        chunk_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        in_dim = embed_dim + proprio_dim + task_dim
        self.in_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, chunk_len * action_dim),
        )

    def forward(self, emb: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb.float(), proprio.float(), task.float()], dim=-1)
        x = self.in_proj(x) + self.pos[:, : x.shape[1]]
        y = self.encoder(x)
        chunk = self.out(y[:, -1])
        return chunk.view(chunk.shape[0], self.chunk_len, self.action_dim)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    timesteps = timesteps.reshape(-1).float()
    half = dim // 2
    if half == 0:
        return timesteps[:, None]
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps[:, None] * freqs[None, :] * (2.0 * math.pi)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class LeWMLiberoFlowMatchingHead(nn.Module):
    """GR00T-style flow-matching action chunk expert conditioned on frozen LeWM embeddings."""

    def __init__(
        self,
        embed_dim: int,
        proprio_dim: int,
        task_dim: int,
        action_dim: int,
        seq_len: int,
        chunk_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        in_dim = embed_dim + proprio_dim + task_dim
        self.cond_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU())
        self.cond_pos = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, chunk_len, hidden_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def _condition(self, emb: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([emb.float(), proprio.float(), task.float()], dim=-1)
        return self.cond_proj(cond) + self.cond_pos[:, : cond.shape[1]]

    def forward(
        self,
        emb: torch.Tensor,
        proprio: torch.Tensor,
        task: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._condition(emb, proprio, task)
        x = self.action_proj(noisy_actions.float()) + self.action_pos[:, : noisy_actions.shape[1]]
        t_emb = sinusoidal_timestep_embedding(timesteps, x.shape[-1]).to(dtype=x.dtype)
        x = x + self.time_mlp(t_emb).unsqueeze(1)
        y = self.decoder(tgt=x, memory=cond)
        return self.out(y)

    def compute_loss(
        self,
        emb: torch.Tensor,
        proprio: torch.Tensor,
        task: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        actions = actions.float()
        noise = torch.randn_like(actions)
        t = torch.rand((actions.shape[0], 1, 1), dtype=torch.float32, device=actions.device)
        noisy_actions = (1.0 - t) * noise + t * actions
        target_velocity = actions - noise
        pred_velocity = self(emb, proprio, task, noisy_actions, t)
        loss = F.mse_loss(pred_velocity.float(), target_velocity)
        action_pred = noisy_actions + (1.0 - t) * pred_velocity.float()
        return {"loss": loss, "pred": action_pred}

    @torch.no_grad()
    def sample_actions(
        self,
        emb: torch.Tensor,
        proprio: torch.Tensor,
        task: torch.Tensor,
        steps: int = 4,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        steps = max(int(steps), 1)
        actions = torch.randn(
            emb.shape[0],
            self.chunk_len,
            self.action_dim,
            dtype=torch.float32,
            device=emb.device,
        ) * float(noise_scale)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((emb.shape[0], 1, 1), step / steps, dtype=torch.float32, device=emb.device)
            velocity = self(emb, proprio, task, actions, t).float()
            actions = actions + dt * velocity
        return actions


class LeWMLiberoActionQueryHead(nn.Module):
    """Deterministic action-query decoder for chunked actions."""

    def __init__(
        self,
        embed_dim: int,
        proprio_dim: int,
        task_dim: int,
        action_dim: int,
        seq_len: int,
        chunk_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        in_dim = embed_dim + proprio_dim + task_dim
        self.cond_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU())
        self.cond_pos = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_len, hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, emb: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([emb.float(), proprio.float(), task.float()], dim=-1)
        memory = self.cond_proj(cond) + self.cond_pos[:, : cond.shape[1]]
        queries = self.action_queries.expand(cond.shape[0], -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.out(decoded)


class LeWMLiberoACTCVAEHead(nn.Module):
    """ACT-style CVAE action chunk expert with deterministic z=0 inference."""

    def __init__(
        self,
        embed_dim: int,
        proprio_dim: int,
        task_dim: int,
        action_dim: int,
        seq_len: int,
        chunk_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        latent_dim: int = 32,
        kl_beta: float = 1e-4,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.kl_beta = kl_beta
        in_dim = embed_dim + proprio_dim + task_dim
        self.cond_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU())
        self.cond_pos = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, chunk_len, hidden_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers // 2))
        self.latent_stats = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim * 2))
        self.latent_to_query = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_len, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def _condition(self, emb: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([emb.float(), proprio.float(), task.float()], dim=-1)
        return self.cond_proj(cond) + self.cond_pos[:, : cond.shape[1]]

    def _encode_latent(self, cond: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_tokens = self.action_proj(actions.float()) + self.action_pos[:, : actions.shape[1]]
        cls = self.cls.expand(actions.shape[0], -1, -1) + cond[:, -1:, :]
        encoded = self.latent_encoder(torch.cat([cls, action_tokens], dim=1))
        stats = self.latent_stats(encoded[:, 0])
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)

    def _decode(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        queries = self.action_queries.expand(cond.shape[0], -1, -1)
        queries = queries + self.latent_to_query(z.float()).unsqueeze(1)
        decoded = self.decoder(tgt=queries, memory=cond)
        return self.out(decoded)

    def forward(self, emb: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        cond = self._condition(emb, proprio, task)
        z = torch.zeros((cond.shape[0], self.latent_dim), dtype=cond.dtype, device=cond.device)
        return self._decode(cond, z)

    def compute_loss(
        self,
        emb: torch.Tensor,
        proprio: torch.Tensor,
        task: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cond = self._condition(emb, proprio, task)
        mu, logvar = self._encode_latent(cond, actions)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        pred = self._decode(cond, z)
        recon = F.l1_loss(pred.float(), actions.float())
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + self.kl_beta * kl
        return {"loss": loss, "pred": pred, "recon": recon.detach(), "kl": kl.detach()}


def build_libero_action_head(head_config: dict[str, Any]) -> nn.Module:
    head_type = str(head_config.get("head_type", "oft"))
    base_keys = {
        "embed_dim",
        "proprio_dim",
        "task_dim",
        "action_dim",
        "seq_len",
        "chunk_len",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "dropout",
    }
    ctor_keys = set(base_keys)
    if head_type == "act_cvae":
        ctor_keys.update({"latent_dim", "kl_beta"})
    kwargs = {key: head_config[key] for key in ctor_keys if key in head_config}
    if head_type == "oft":
        return LeWMLiberoOFTHead(**kwargs)
    if head_type == "groot_fm":
        return LeWMLiberoFlowMatchingHead(**kwargs)
    if head_type == "action_query":
        return LeWMLiberoActionQueryHead(**kwargs)
    if head_type == "act_cvae":
        return LeWMLiberoACTCVAEHead(**kwargs)
    raise ValueError(f"Unknown LIBERO action head type: {head_type}")


class LeWMLiberoOFTPolicy(nn.Module):
    def __init__(self, world_model: nn.Module, head: nn.Module):
        super().__init__()
        self.world_model = world_model
        self.head = head

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 5:
            return self.world_model.encode({"pixels": pixels})["emb"]
        if pixels.ndim != 6:
            raise ValueError(f"Expected pixels with 5 or 6 dims, got {tuple(pixels.shape)}")
        batch, seq_len, n_views, channels, height, width = pixels.shape
        flat = pixels.permute(0, 2, 1, 3, 4, 5).reshape(batch * n_views, seq_len, channels, height, width)
        emb = self.world_model.encode({"pixels": flat})["emb"]
        return emb.reshape(batch, n_views, seq_len, emb.shape[-1]).permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_pixels(pixels), proprio, task)


def save_checkpoint(
    policy: LeWMLiberoOFTPolicy,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    epoch: int,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    head_config: dict[str, Any],
    dataset: LiberoGoalLeWMOFTDataset,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"lewm_libero_oft_head_epoch_{epoch}.ckpt"
    policy_cpu = policy.cpu()
    torch.save(
        {
            "world_model": policy_cpu.world_model,
            "head_state_dict": policy_cpu.head.state_dict(),
            "head_config": head_config,
            "action_scaler": dataset.action_scaler.state_dict(),
            "proprio_scaler": dataset.proprio_scaler.state_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
            "history": history,
        },
        model_path,
    )
    policy.to(args._device)
    torch.save(
        {
            "epoch": epoch,
            "head_state_dict": policy.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "head_config": head_config,
        },
        run_dir / "training_state.pt",
    )
    (run_dir / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")}, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(args.libero_root / ".libero"))
    prepend(args.libero_root)
    prepend(LEWM_CODE)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args._device = device
    transform = make_image_transform(args.img_size)
    image_keys = parse_image_keys(args.image_keys)
    prepend(args.libero_root)
    from libero.libero import benchmark  # noqa: WPS433

    bench = benchmark.get_benchmark_dict()["libero_goal"]()
    task_ids = parse_task_ids(args.tasks, bench.n_tasks)
    task_repeat = parse_task_repeat(args.task_repeat)
    bad_repeats = [task_id for task_id in task_repeat if task_id not in task_ids]
    if bad_repeats:
        raise ValueError(f"--task-repeat includes tasks outside --tasks: {bad_repeats}")
    dataset = LiberoGoalLeWMOFTDataset(
        args.libero_root,
        transform,
        args.img_size,
        args.seq_len,
        args.chunk_len,
        image_keys,
        task_ids,
        args.center_crop,
        args.rotate_images_180,
        args.filter_noop,
        args.noop_threshold,
        args.normalization,
        args.normalize_gripper_action,
        task_repeat,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=generator,
    )

    resume_payload: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        world_model = resume_payload["world_model"].to(device)
        resume_head_type = str(resume_payload.get("head_config", {}).get("head_type", "oft"))
        if resume_head_type != args.head_type:
            raise ValueError(f"Cannot resume {resume_head_type!r} head checkpoint with --head-type {args.head_type!r}")
    else:
        world_model = load_world_model(
            args.init_policy,
            device,
            encoder_kind=args.encoder_kind,
            dino_model_name=args.dino_model,
        )
    world_model.train(args.train_encoder)
    world_model.requires_grad_(args.train_encoder)
    base_embed_dim = int(world_model.predictor.pos_embedding.shape[-1])
    embed_dim = base_embed_dim * len(image_keys)
    head_config = {
        "head_type": args.head_type,
        "embed_dim": embed_dim,
        "proprio_dim": 8,
        "task_dim": 10,
        "action_dim": 7,
        "seq_len": args.seq_len,
        "chunk_len": args.chunk_len,
        "action_horizon": args.action_horizon,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "fm_sampling_steps": args.fm_sampling_steps,
        "latent_dim": args.latent_dim,
        "kl_beta": args.kl_beta,
        "image_keys": image_keys,
        "task_ids": task_ids,
        "center_crop": bool(args.center_crop),
        "rotate_images_180": bool(args.rotate_images_180),
        "action_convention": "openvla_oft_train_gripper_1_open_0_close",
        "action_normalization_mask": dataset.action_scaler.state_dict()["mask"].astype(bool).tolist(),
        "proprio_convention": "ee_axis_angle_6_plus_2d_gripper",
    }
    head = build_libero_action_head(head_config)
    if resume_payload is not None:
        head.load_state_dict(resume_payload["head_state_dict"])
    policy = LeWMLiberoOFTPolicy(world_model, head).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and args.precision == "bf16"

    print(
        json.dumps(
            {
                "dataset_len": len(dataset),
                "seq_len": args.seq_len,
                "chunk_len": args.chunk_len,
                "action_horizon": args.action_horizon,
                "task_ids": task_ids,
                "image_keys": image_keys,
                "init_policy": str(args.init_policy),
                "encoder_kind": args.encoder_kind,
                "dino_model": args.dino_model,
                "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
                "run_dir": str(args.run_dir),
                "device": str(device),
                "train_encoder": bool(args.train_encoder),
                "trainable_parameters": int(sum(p.numel() for p in trainable)),
                "normalization": args.normalization,
                "normalize_gripper_action": bool(args.normalize_gripper_action),
                "filter_noop": bool(args.filter_noop),
                "task_repeat": task_repeat,
            },
            indent=2,
        ),
        flush=True,
    )

    history: list[dict[str, float]] = []
    epoch_offset = infer_epoch_offset(args.resume_checkpoint) if args.epoch_offset < 0 else args.epoch_offset
    global_start = time.time()
    for local_epoch in range(1, args.max_epochs + 1):
        epoch = epoch_offset + local_epoch
        policy.head.train()
        policy.world_model.train(args.train_encoder)
        totals = {"loss": 0.0, "mae": 0.0, "first_mae": 0.0}
        num_batches = 0
        epoch_start = time.time()
        for batch_idx, batch in enumerate(loader, start=1):
            pixels = batch["pixels"].to(device, non_blocking=True)
            proprios = batch["proprios"].to(device, non_blocking=True)
            tasks = batch["tasks"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                if args.train_encoder:
                    emb = policy.encode_pixels(pixels)
                else:
                    with torch.no_grad():
                        emb = policy.encode_pixels(pixels).float()
                if hasattr(policy.head, "compute_loss"):
                    loss_result = policy.head.compute_loss(emb, proprios, tasks, actions)
                    loss = loss_result["loss"]
                    pred = loss_result["pred"]
                else:
                    pred = policy.head(emb, proprios, tasks)
                    if args.loss == "smooth_l1":
                        loss = torch.nn.functional.smooth_l1_loss(pred.float(), actions.float())
                    else:
                        loss = torch.nn.functional.l1_loss(pred.float(), actions.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            with torch.no_grad():
                mae = (pred.float() - actions.float()).abs().mean()
                first_mae = (pred[:, 0].float() - actions[:, 0].float()).abs().mean()
            totals["loss"] += float(loss.detach().cpu())
            totals["mae"] += float(mae.detach().cpu())
            totals["first_mae"] += float(first_mae.detach().cpu())
            num_batches += 1
            if batch_idx % 50 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss": totals["loss"] / num_batches,
                            "mae": totals["mae"] / num_batches,
                            "first_mae": totals["first_mae"] / num_batches,
                            "elapsed_sec": time.time() - global_start,
                        }
                    ),
                    flush=True,
                )
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
        row = {
            "epoch": epoch,
            "local_epoch": local_epoch,
            "batches": num_batches,
            "loss": totals["loss"] / max(num_batches, 1),
            "mae": totals["mae"] / max(num_batches, 1),
            "first_mae": totals["first_mae"] / max(num_batches, 1),
            "epoch_sec": time.time() - epoch_start,
            "elapsed_sec": time.time() - global_start,
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        if local_epoch % args.save_every == 0 or local_epoch == args.max_epochs:
            save_checkpoint(policy, optimizer, args.run_dir, epoch, args, history, head_config, dataset)


if __name__ == "__main__":
    main()
