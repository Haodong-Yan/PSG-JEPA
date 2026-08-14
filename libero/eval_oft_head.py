#!/usr/bin/env python3
"""Evaluate a LeWM + OFT-style LIBERO action chunk head."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
# Benchmark asset tree (bddl files + init states). Overridable with --libero-root or
# $LIBERO_ROOT; the default matches the clone location the README describes.
LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT") or ROOT / "assets" / "benchmarks" / "LIBERO")
LEWM_CODE = ROOT
SCRIPTS_DIR = ROOT / "libero"


def prepend(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def parse_task_ids(value: str) -> list[int]:
    if value.lower() == "all":
        return list(range(10))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT,
                        help="LIBERO benchmark asset tree (bddl files + init states)")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--n-eval", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--env-img-res", type=int, default=128)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--action-horizon-override", type=int, default=0)
    parser.add_argument("--fm-sampling-steps-override", type=int, default=0)
    parser.add_argument("--fm-noise-scale", type=float, default=1.0)
    parser.add_argument("--clip-normalized-actions", action="store_true")
    return parser.parse_args()


def make_cfg(args: argparse.Namespace, image_keys: list[str]) -> EasyDict:
    obs_key_mapping = {
        "agentview_rgb": "agentview_image",
        "eye_in_hand_rgb": "robot0_eye_in_hand_image",
        "ee_pos": "robot0_eef_pos",
        "ee_quat": "robot0_eef_quat",
        "gripper_states": "robot0_gripper_qpos",
    }
    return EasyDict(
        {
            "device": args.device,
            "bddl_folder": str(args.libero_root / "libero" / "libero" / "bddl_files"),
            "init_states_folder": str(args.libero_root / "libero" / "libero" / "init_files"),
            "data": {
                "img_h": args.env_img_res,
                "img_w": args.env_img_res,
                "obs": {
                    "modality": {
                        "rgb": image_keys,
                        "depth": [],
                        "low_dim": ["ee_pos", "ee_quat", "gripper_states"],
                    }
                },
                "obs_key_mapping": obs_key_mapping,
            },
            "eval": {
                "n_eval": args.n_eval,
                "max_steps": args.max_steps,
                "num_procs": max(1, min(args.num_procs, args.n_eval)),
                "use_mp": args.num_procs > 1,
                "save_sim_states": False,
            },
            "lifelong": {"algo": "LeWMOFTHead"},
        }
    )


def build_task_onehot(task_id: int, batch: int, device: torch.device) -> torch.Tensor:
    task = torch.zeros((batch, 10), dtype=torch.float32, device=device)
    task[:, task_id] = 1.0
    return task


def scaler_inverse(state: dict[str, Any], values: torch.Tensor) -> torch.Tensor:
    kind = state.get("kind")
    mask = torch.as_tensor(state.get("mask", np.ones(values.shape[-1], dtype=bool)), dtype=torch.bool, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(0)
    if kind == "zscore":
        mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=values.device)
        std = torch.as_tensor(state["std"], dtype=torch.float32, device=values.device)
        restored = values.float() * std + mean
        return torch.where(mask, restored, values.float())
    if kind == "bounds_q99":
        low = torch.as_tensor(state["low"], dtype=torch.float32, device=values.device)
        high = torch.as_tensor(state["high"], dtype=torch.float32, device=values.device)
        span = torch.clamp(high - low, min=1e-6)
        restored = (values.float() + 1.0) * 0.5 * span + low
        return torch.where(mask, restored, values.float())
    raise ValueError(f"Unknown scaler kind: {kind}")


def scaler_transform(state: dict[str, Any], values: torch.Tensor) -> torch.Tensor:
    kind = state.get("kind")
    mask = torch.as_tensor(state.get("mask", np.ones(values.shape[-1], dtype=bool)), dtype=torch.bool, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(0)
    if kind == "zscore":
        mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=values.device)
        std = torch.as_tensor(state["std"], dtype=torch.float32, device=values.device)
        normalized = (values.float() - mean) / std
        return torch.where(mask, normalized, values.float())
    if kind == "bounds_q99":
        low = torch.as_tensor(state["low"], dtype=torch.float32, device=values.device)
        high = torch.as_tensor(state["high"], dtype=torch.float32, device=values.device)
        span = torch.clamp(high - low, min=1e-6)
        normalized = torch.clamp(2.0 * (values.float() - low) / span - 1.0, -1.0, 1.0)
        return torch.where(mask, normalized, values.float())
    raise ValueError(f"Unknown scaler kind: {kind}")


def quat_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.float()
    xyz = quat[..., :3]
    w = quat[..., 3:4].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    angle = 2.0 * torch.acos(w)
    safe = den > 1e-6
    return torch.where(safe, xyz * angle / torch.clamp(den, min=1e-6), torch.zeros_like(xyz))


def openvla_train_action_to_env(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    gripper = np.clip(action[..., -1], 0.0, 1.0)
    gripper = np.where(gripper >= 0.5, 1.0, -1.0)  # [0,1] -> [-1,+1]
    action[..., -1] = -gripper  # 1=open -> -1 env open; 0=close -> +1 env close
    return np.clip(action, -1.0, 1.0)


class EvalLeWMLiberoOFTPolicy(torch.nn.Module):
    def __init__(self, payload: dict[str, Any], device: torch.device):
        super().__init__()
        prepend(SCRIPTS_DIR)
        prepend(LEWM_CODE)
        from train_oft_head import build_libero_action_head  # noqa: WPS433

        self.world_model = payload["world_model"]
        head_config = dict(payload["head_config"])
        self.seq_len = int(head_config["seq_len"])
        self.chunk_len = int(head_config["chunk_len"])
        self.head_type = str(head_config.get("head_type", "oft"))
        self.action_horizon = int(head_config.get("action_horizon", self.chunk_len))
        self.fm_sampling_steps = int(head_config.get("fm_sampling_steps", 4))
        self.fm_noise_scale = 1.0
        self.clip_normalized_actions = False
        self.image_keys = list(head_config.get("image_keys", ["agentview_rgb"]))
        self.trained_task_ids = list(head_config.get("task_ids", list(range(10))))
        self.center_crop = bool(head_config.get("center_crop", True))
        self.rotate_images_180 = bool(head_config.get("rotate_images_180", True))
        self.head = build_libero_action_head(head_config)
        self.head.load_state_dict(payload["head_state_dict"])
        self.action_scaler = payload["action_scaler"]
        self.proprio_scaler = payload["proprio_scaler"]
        self.task_id = 0
        self.pixel_history: dict[str, deque[torch.Tensor]] | None = None
        self.proprio_history: deque[torch.Tensor] | None = None
        self.action_queue: deque[np.ndarray] = deque()
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.to(device)
        self.eval()
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset(self) -> None:
        self.pixel_history = None
        self.proprio_history = None
        self.action_queue.clear()

    def set_task_id(self, task_id: int) -> None:
        self.task_id = task_id
        self.reset()

    def _prep_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(self.device).float()
        if pixels.max() > 2.0:
            pixels = pixels / 255.0
        if self.rotate_images_180:
            pixels = torch.flip(pixels, dims=(-2, -1))
        if self.center_crop:
            height, width = pixels.shape[-2:]
            crop_h, crop_w = int(math.floor(height * 0.9)), int(math.floor(width * 0.9))
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            pixels = pixels[..., top : top + crop_h, left : left + crop_w]
        if pixels.shape[-2:] != (224, 224):
            pixels = F.interpolate(pixels, size=(224, 224), mode="bilinear", align_corners=False)
        return (pixels - self.image_mean) / self.image_std

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 5:
            return self.world_model.encode({"pixels": pixels})["emb"]
        if pixels.ndim != 6:
            raise ValueError(f"Expected pixels with 5 or 6 dims, got {tuple(pixels.shape)}")
        batch, seq_len, n_views, channels, height, width = pixels.shape
        flat = pixels.permute(0, 2, 1, 3, 4, 5).reshape(batch * n_views, seq_len, channels, height, width)
        emb = self.world_model.encode({"pixels": flat})["emb"]
        return emb.reshape(batch, n_views, seq_len, emb.shape[-1]).permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

    def _plan_chunk(self) -> None:
        if self.pixel_history is None or self.proprio_history is None:
            raise RuntimeError("Observation history is not initialized")
        stacked_pixels = torch.stack(
            [torch.stack(list(self.pixel_history[key]), dim=1) for key in self.image_keys],
            dim=2,
        )
        stacked_proprio = torch.stack(list(self.proprio_history), dim=1)
        batch = stacked_proprio.shape[0]
        if batch != 1:
            raise ValueError("OFT-head evaluator currently expects num_procs=1 / batch size 1")
        task = build_task_onehot(self.task_id, batch, self.device).unsqueeze(1).expand(-1, self.seq_len, -1)
        with torch.no_grad():
            emb = self._encode_pixels(stacked_pixels)
            if hasattr(self.head, "sample_actions"):
                normalized = self.head.sample_actions(
                    emb,
                    stacked_proprio,
                    task,
                    steps=self.fm_sampling_steps,
                    noise_scale=self.fm_noise_scale,
                )
            else:
                normalized = self.head(emb, stacked_proprio, task)
            if self.clip_normalized_actions:
                normalized = normalized.clamp(-1.0, 1.0)
            action = scaler_inverse(self.action_scaler, normalized)
        actions = openvla_train_action_to_env(action[0].detach().cpu().numpy())
        self.action_queue.clear()
        for action_row in actions[: self.action_horizon]:
            self.action_queue.append(action_row.astype(np.float32, copy=False))

    def get_action(self, data: dict[str, Any]) -> np.ndarray:
        current_pixels = {key: self._prep_pixels(data["obs"][key]) for key in self.image_keys}
        batch = next(iter(current_pixels.values())).shape[0]
        ee_pos = data["obs"]["ee_pos"].to(self.device).float()
        ee_quat = data["obs"]["ee_quat"].to(self.device).float()
        gripper = data["obs"]["gripper_states"].to(self.device).float()
        proprio = torch.cat([ee_pos, quat_to_axis_angle(ee_quat), gripper], dim=-1)
        proprio = scaler_transform(self.proprio_scaler, proprio)
        if self.pixel_history is None:
            self.pixel_history = {key: deque([value] * self.seq_len, maxlen=self.seq_len) for key, value in current_pixels.items()}
            self.proprio_history = deque([proprio] * self.seq_len, maxlen=self.seq_len)
        else:
            for key, value in current_pixels.items():
                self.pixel_history[key].append(value)
            self.proprio_history.append(proprio)
        if batch != 1:
            raise ValueError("OFT-head evaluator currently expects num_procs=1 / batch size 1")
        if not self.action_queue:
            self._plan_chunk()
        return self.action_queue.popleft()[None, :]


class AlgoWrapper(torch.nn.Module):
    def __init__(self, policy: EvalLeWMLiberoOFTPolicy):
        super().__init__()
        self.policy = policy

    def reset(self) -> None:
        self.policy.reset()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(args.libero_root / ".libero"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepend(args.libero_root)
    prepend(LEWM_CODE)
    prepend(SCRIPTS_DIR)

    from libero.libero import benchmark
    from libero.lifelong.metric import evaluate_one_task_success
    import robomimic.utils.obs_utils as ObsUtils

    import compat  # noqa: WPS433 -- must run before unpickling, see compat.py

    compat.install()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = EvalLeWMLiberoOFTPolicy(payload, device)
    train_args = dict(payload.get("args", {}))
    train_encoder = bool(train_args.get("train_encoder", False))
    if args.action_horizon_override:
        policy.action_horizon = max(1, min(int(args.action_horizon_override), policy.chunk_len))
    if args.fm_sampling_steps_override:
        policy.fm_sampling_steps = max(1, int(args.fm_sampling_steps_override))
    policy.fm_noise_scale = float(args.fm_noise_scale)
    policy.clip_normalized_actions = bool(args.clip_normalized_actions)
    algo = AlgoWrapper(policy)
    cfg = make_cfg(args, policy.image_keys)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
    bench = benchmark.get_benchmark_dict()["libero_goal"]()
    task_ids = parse_task_ids(args.tasks)

    started = time.time()
    rows = []
    for task_id in task_ids:
        policy.set_task_id(task_id)
        task = bench.get_task(task_id)
        task_emb = torch.zeros((1, 1), dtype=torch.float32, device=device)
        task_start = time.time()
        success_rate = evaluate_one_task_success(cfg, algo, task, task_emb, task_id)
        rows.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "language": task.language,
                "success_rate": float(success_rate),
                "n_eval": args.n_eval,
                "elapsed_sec": time.time() - task_start,
            }
        )
        print(f"[libero-lewm-oft-head] task={task_id} success={success_rate:.3f}", flush=True)

    result = {
        "classification": (
            f"rcaux_libero_goal_lewm_trainable_encoder_{policy.head_type}_head_official_eval"
            if train_encoder
            else f"rcaux_libero_goal_lewm_frozen_encoder_{policy.head_type}_head_official_eval"
        ),
        "checkpoint": str(args.checkpoint),
        "head_type": policy.head_type,
        "fm_sampling_steps": policy.fm_sampling_steps if hasattr(policy.head, "sample_actions") else None,
        "fm_noise_scale": policy.fm_noise_scale if hasattr(policy.head, "sample_actions") else None,
        "clip_normalized_actions": policy.clip_normalized_actions,
        "train_encoder": train_encoder,
        "trained_task_ids": policy.trained_task_ids,
        "chunk_len": policy.chunk_len,
        "action_horizon": policy.action_horizon,
        "task_ids": task_ids,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "env_img_res": args.env_img_res,
        "mean_success_rate": float(np.mean([row["success_rate"] for row in rows])),
        "tasks": rows,
        "elapsed_sec": time.time() - started,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if args.output is None:
        label = "all" if len(task_ids) == 10 else "_".join(map(str, task_ids))
        args.output = ROOT / "results" / f"libero_goal_rcaux_lewm_oft_head_eval_tasks_{label}_n{args.n_eval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
