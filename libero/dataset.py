#!/usr/bin/env python3
"""LIBERO-Goal demo dataset + encoder loading shared by the OFT head train/eval scripts.

Running this file directly trains the simple MLP BC head (`LeWMLiberoBCHead`); the paper's
policy-learning numbers use the OFT head in ``train_oft_head.py``, which imports the dataset,
the encoder loaders and the image transform from here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import tv_tensors


ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = ROOT / "assets" / "benchmarks" / "LIBERO"
LEWM_CODE = ROOT
SCRIPTS_DIR = ROOT / "libero"
# No default encoder checkpoint: a fresh clone has none. Pass --init-policy with the encoder
# produced by pretraining, or downloaded via weights/download_weights.py.
DEFAULT_LEWM = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument("--init-policy", type=Path, default=DEFAULT_LEWM)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "libero_goal_lewm_bc",
    )
    parser.add_argument("--n-obs-steps", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--train-encoder", action="store_true")
    return parser.parse_args()


def prepend(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def parse_image_keys(value: str) -> list[str]:
    keys = [part.strip() for part in value.split(",") if part.strip()]
    if not keys:
        raise ValueError("--image-keys must contain at least one observation key")
    return keys


def parse_task_ids(value: str, n_tasks: int) -> list[int]:
    if value.lower() == "all":
        return list(range(n_tasks))
    task_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not task_ids:
        raise ValueError("--tasks must be 'all' or contain at least one task id")
    bad = [task_id for task_id in task_ids if task_id < 0 or task_id >= n_tasks]
    if bad:
        raise ValueError(f"Task ids out of range for LIBERO-Goal: {bad}")
    return task_ids


def find_model_with_attr(obj: Any, attr: str) -> Any | None:
    if hasattr(obj, attr) and hasattr(obj, "parameters"):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = find_model_with_attr(value, attr)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_model_with_attr(value, attr)
            if found is not None:
                return found
    return None


def load_world_model(path: Path, device: torch.device) -> nn.Module:
    prepend(LEWM_CODE)
    import compat  # noqa: WPS433 -- must run before unpickling, see compat.py

    compat.install()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = find_model_with_attr(payload, "encode")
    if model is None:
        raise RuntimeError(f"Could not find LeWM model with encode() in {path}")
    return model.to(device)


def make_image_transform(img_size: int):
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(size=(img_size, img_size), antialias=True),
        ]
    )


def demo_sort_key(name: str) -> int:
    match = re.fullmatch(r"demo_(\d+)", name)
    return int(match.group(1)) if match else 0


def demo_eval_lowdim(demo: h5py.Group, indices: Any) -> np.ndarray:
    """Low-dim observation in the same order used by the eval wrapper."""
    joint = np.asarray(demo["obs"]["joint_states"][indices], dtype=np.float32)
    gripper = np.asarray(demo["obs"]["gripper_states"][indices], dtype=np.float32)
    return np.concatenate([joint, gripper], axis=-1)


class SafeScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32, copy=False)
        self.std = np.maximum(std.astype(np.float32, copy=False), 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std + self.mean


class LiberoGoalLeWMDataset(torch.utils.data.Dataset):
    def __init__(self, libero_root: Path, transform: Any, img_size: int, n_obs_steps: int):
        self.libero_root = libero_root
        self.transform = transform
        self.img_size = img_size
        self.n_obs_steps = n_obs_steps
        self.files: list[Path] = []
        self.samples: list[tuple[int, int, str, int]] = []
        self._handles: dict[int, h5py.File] = {}

        prepend(libero_root)
        from libero.libero import benchmark  # noqa: WPS433

        bench = benchmark.get_benchmark_dict()["libero_goal"]()
        for task_id in range(bench.n_tasks):
            path = libero_root / "datasets" / bench.get_task_demonstration(task_id)
            file_idx = len(self.files)
            self.files.append(path)
            with h5py.File(path, "r") as handle:
                demos = handle["data"]
                for demo_name in sorted(demos.keys(), key=demo_sort_key):
                    length = int(demos[demo_name]["actions"].shape[0])
                    first = max(0, self.n_obs_steps - 1)
                    for step in range(first, length):
                        self.samples.append((task_id, file_idx, demo_name, step))

        actions, proprios = [], []
        for path in self.files:
            with h5py.File(path, "r") as handle:
                for demo_name in sorted(handle["data"].keys(), key=demo_sort_key):
                    demo = handle["data"][demo_name]
                    actions.append(np.asarray(demo["actions"], dtype=np.float32))
                    proprios.append(demo_eval_lowdim(demo, slice(None)))
        action_arr = np.concatenate(actions, axis=0)
        proprio_arr = np.concatenate(proprios, axis=0)
        self.action_scaler = SafeScaler(action_arr.mean(0, keepdims=True), action_arr.std(0, keepdims=True))
        self.proprio_scaler = SafeScaler(proprio_arr.mean(0, keepdims=True), proprio_arr.std(0, keepdims=True))

    def __len__(self) -> int:
        return len(self.samples)

    def _handle(self, file_idx: int) -> h5py.File:
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self.files[file_idx], "r")
        return self._handles[file_idx]

    def _prep_image(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = np.asarray(Image.fromarray(image).resize((self.img_size, self.img_size), Image.BILINEAR))
        image = np.transpose(image.astype(np.uint8, copy=False), (2, 0, 1))
        return self.transform(tv_tensors.Image(image))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task_id, file_idx, demo_name, step = self.samples[index]
        demo = self._handle(file_idx)["data"][demo_name]
        obs = demo["obs"]["agentview_rgb"]
        first = 0
        obs_indices = [max(first, step - self.n_obs_steps + 1 + i) for i in range(self.n_obs_steps)]
        pixels = torch.stack([self._prep_image(np.asarray(obs[i])) for i in obs_indices], dim=0)
        action = self.action_scaler.transform(np.asarray(demo["actions"][step], dtype=np.float32))
        proprio = self.proprio_scaler.transform(demo_eval_lowdim(demo, step))
        task = np.zeros((10,), dtype=np.float32)
        task[task_id] = 1.0
        return {
            "pixels": pixels,
            "action": torch.from_numpy(action.reshape(-1).astype(np.float32)),
            "proprio": torch.from_numpy(proprio.reshape(-1).astype(np.float32)),
            "task": torch.from_numpy(task),
        }


class LeWMLiberoBCHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_obs_steps: int,
        proprio_dim: int,
        task_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * n_obs_steps + proprio_dim + task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, emb: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb.flatten(1).float(), proprio.float(), task.float()], dim=-1)
        return self.net(x)


class LeWMLiberoBCPolicy(nn.Module):
    def __init__(self, world_model: nn.Module, head: LeWMLiberoBCHead):
        super().__init__()
        self.world_model = world_model
        self.head = head

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        emb = self.world_model.encode({"pixels": pixels})["emb"]
        return self.head(emb, proprio, task)


def save_checkpoint(
    policy: LeWMLiberoBCPolicy,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    epoch: int,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    head_config: dict[str, Any],
    dataset: LiberoGoalLeWMDataset,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"lewm_libero_bc_epoch_{epoch}.ckpt"
    policy_cpu = policy.cpu()
    torch.save(
        {
            "world_model": policy_cpu.world_model,
            "head_state_dict": policy_cpu.head.state_dict(),
            "head_config": head_config,
            "action_mean": dataset.action_scaler.mean,
            "action_std": dataset.action_scaler.std,
            "proprio_mean": dataset.proprio_scaler.mean,
            "proprio_std": dataset.proprio_scaler.std,
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
    dataset = LiberoGoalLeWMDataset(args.libero_root, transform, args.img_size, args.n_obs_steps)
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

    world_model = load_world_model(args.init_policy, device)
    world_model.train(args.train_encoder)
    world_model.requires_grad_(args.train_encoder)
    embed_dim = int(world_model.predictor.pos_embedding.shape[-1])
    head_config = {
        "embed_dim": embed_dim,
        "n_obs_steps": args.n_obs_steps,
        "proprio_dim": 9,
        "task_dim": 10,
        "action_dim": 7,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    head = LeWMLiberoBCHead(**head_config)
    policy = LeWMLiberoBCPolicy(world_model, head).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and args.precision == "bf16"

    print(
        json.dumps(
            {
                "dataset_len": len(dataset),
                "init_policy": str(args.init_policy),
                "run_dir": str(args.run_dir),
                "device": str(device),
                "train_encoder": bool(args.train_encoder),
                "trainable_parameters": int(sum(p.numel() for p in trainable)),
            },
            indent=2,
        ),
        flush=True,
    )

    history: list[dict[str, float]] = []
    global_start = time.time()
    for epoch in range(1, args.max_epochs + 1):
        policy.head.train()
        policy.world_model.train(args.train_encoder)
        totals = {"loss": 0.0, "mae": 0.0}
        num_batches = 0
        epoch_start = time.time()
        for batch_idx, batch in enumerate(loader, start=1):
            pixels = batch["pixels"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            task = batch["task"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                if args.train_encoder:
                    pred = policy(pixels, proprio, task)
                else:
                    with torch.no_grad():
                        emb = policy.world_model.encode({"pixels": pixels})["emb"].float()
                    pred = policy.head(emb, proprio, task)
                loss = torch.nn.functional.smooth_l1_loss(pred.float(), action.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            with torch.no_grad():
                mae = (pred.float() - action.float()).abs().mean()
            totals["loss"] += float(loss.detach().cpu())
            totals["mae"] += float(mae.detach().cpu())
            num_batches += 1
            if batch_idx % 50 == 0:
                print(json.dumps({"epoch": epoch, "batch": batch_idx, "loss": totals["loss"] / num_batches, "mae": totals["mae"] / num_batches, "elapsed_sec": time.time() - global_start}), flush=True)
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
        row = {
            "epoch": epoch,
            "batches": num_batches,
            "loss": totals["loss"] / max(num_batches, 1),
            "mae": totals["mae"] / max(num_batches, 1),
            "epoch_sec": time.time() - epoch_start,
            "elapsed_sec": time.time() - global_start,
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        if epoch % args.save_every == 0 or epoch == args.max_epochs:
            save_checkpoint(policy, optimizer, args.run_dir, epoch, args, history, head_config, dataset)


if __name__ == "__main__":
    main()
