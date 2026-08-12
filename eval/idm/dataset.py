"""Embedding extraction and dataset for IDM training."""

from __future__ import annotations

import os
import time
from pathlib import Path
import h5py
try:
    import hdf5plugin  # registers modern blosc filters in-process
except ImportError:
    pass
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingTripleDataset(Dataset):
    """Dataset of (z_t, z_goal, steps_remaining, a_t) for GC-IDM training."""

    def __init__(
        self,
        embeddings_path: str,
        max_goal_horizon: int = 50,
        frameskip: int = 1,
        train_split: float = 1.0,
        split_seed: int = 42,
        split_partition: str = "train",
    ):
        """
        Args:
            embeddings_path: path to .npz file from extract_embeddings()
            max_goal_horizon: maximum horizon for goal sampling
            frameskip: number of raw steps between paired frames.
            train_split: fraction of episodes to use for training (1.0 = all).
                Set to 0.9 to hold out 10% of episodes.
            split_seed: RNG seed for episode-level split (default 42, matches paper).
            split_partition: "train" or "val" partition to use.
        """
        data = np.load(embeddings_path)
        self.embeddings = data["embeddings"]    # (N, D)
        self.actions = data["actions"]          # (N, act_dim)
        self.ep_ids = data["episode_ids"]       # (N,)
        self.max_goal_horizon = max_goal_horizon
        self.frameskip = frameskip

        # Episode-level split: filter frames to train/val episodes
        if train_split < 1.0:
            unique_eps = np.unique(self.ep_ids)
            n_holdout = max(1, round(len(unique_eps) * (1.0 - train_split)))
            rng = np.random.default_rng(split_seed)
            holdout_eps = set(rng.choice(unique_eps, size=n_holdout, replace=False).tolist())
            if split_partition == "train":
                keep_eps = set(unique_eps.tolist()) - holdout_eps
            else:
                keep_eps = holdout_eps
            ep_mask = np.isin(self.ep_ids, list(keep_eps))
            self.held_out_episodes = sorted(holdout_eps)
            print(f"  Episode split: {len(keep_eps)}/{len(unique_eps)} episodes "
                  f"({ep_mask.sum()}/{len(ep_mask)} frames, partition={split_partition})")
            print(f"  Held-out episode IDs ({len(self.held_out_episodes)}): {self.held_out_episodes}")
        else:
            ep_mask = np.ones(len(self.ep_ids), dtype=bool)
            self.held_out_episodes = []

        # Build valid pair indices
        n = len(self.ep_ids)
        same_ep = self.ep_ids[:-frameskip] == self.ep_ids[frameskip:]
        if frameskip == 1:
            no_nan = ~np.isnan(self.actions[:-frameskip]).any(axis=1)
        else:
            nan_mask = np.isnan(self.actions).any(axis=1)
            kernel = np.ones(frameskip, dtype=bool)
            has_nan = np.convolve(nan_mask, kernel, mode='valid')[:n - frameskip] > 0
            no_nan = ~has_nan
        valid = same_ep & no_nan & ep_mask[:-frameskip]
        self.valid_indices: list[int] = np.nonzero(valid)[0].tolist()

        self._build_episode_map()

    def _build_episode_map(self):
        """Map each episode ID to its frame range for goal sampling."""
        self.ep_ranges: dict[int, tuple[int, int]] = {}
        current_ep = self.ep_ids[0]
        start = 0
        for i in range(1, len(self.ep_ids)):
            if self.ep_ids[i] != current_ep:
                self.ep_ranges[current_ep] = (start, i)
                current_ep = self.ep_ids[i]
                start = i
        self.ep_ranges[current_ep] = (start, len(self.ep_ids))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        i = self.valid_indices[idx]
        k = self.frameskip
        z_t = torch.from_numpy(self.embeddings[i]).float()
        a_t = torch.from_numpy(self.actions[i : i + k].reshape(-1)).float()

        # Sample a future frame as goal
        ep = self.ep_ids[i]
        ep_start, ep_end = self.ep_ranges[ep]
        max_future = min(i + self.max_goal_horizon, ep_end - 1)

        if max_future <= i + 1:
            goal_idx = i + 1
        else:
            goal_idx = np.random.randint(i + 1, max_future + 1)

        z_goal = torch.from_numpy(self.embeddings[goal_idx]).float()
        steps_remaining = torch.tensor(goal_idx - i, dtype=torch.long)

        return {
            "z_t": z_t,
            "z_goal": z_goal,
            "steps_remaining": steps_remaining,
            "action": a_t,
        }


def load_lewm_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LeWM model from any supported format.

    Supports:
      1. *_object.ckpt — torch.save'd full object (standard le-wm output)
      2. Directory with config.json + weights.pt (HuggingFace download)
      3. HuggingFace repo ID like 'quentinll/lewm-tworooms' (auto-downloads)

    Returns the raw model (LeWM or spt.Module wrapper), moved to device.
    """
    from pathlib import Path

    path = Path(checkpoint_path)

    # Case 1: _object.ckpt file
    if path.is_file() and path.suffix == ".ckpt":
        print(f"Loading _object.ckpt: {path}")
        model = torch.load(str(path), map_location=device, weights_only=False)
        model.eval()
        return model

    # Case 2: directory with config.json + weights.pt
    if path.is_dir():
        cfg_file = path / "config.json"
        wts_file = path / "weights.pt"
        if cfg_file.exists() and wts_file.exists():
            print(f"Loading from config.json + weights.pt in {path}")
            return _load_from_config_weights(str(cfg_file), str(wts_file), device)

    # Case 3: HuggingFace repo ID (contains '/')
    if "/" in checkpoint_path and not path.exists():
        print(f"Downloading from HuggingFace: {checkpoint_path}")
        from huggingface_hub import hf_hub_download
        cfg_file = hf_hub_download(checkpoint_path, "config.json")
        wts_file = hf_hub_download(checkpoint_path, "weights.pt")
        return _load_from_config_weights(cfg_file, wts_file, device)

    # Case 4: maybe it's a glob pattern or directory with _object.ckpt inside
    if path.is_dir():
        ckpts = sorted(path.glob("*_object.ckpt"))
        if ckpts:
            print(f"Loading latest _object.ckpt from {path}: {ckpts[-1].name}")
            model = torch.load(str(ckpts[-1]), map_location=device, weights_only=False)
            model.eval()
            return model

    raise FileNotFoundError(
        f"Cannot load checkpoint from '{checkpoint_path}'. "
        f"Expected: _object.ckpt file, dir with config.json+weights.pt, or HF repo ID."
    )


def _load_from_config_weights(
    cfg_path: str, wts_path: str, device: str
) -> torch.nn.Module:
    """Reconstruct a LeWM model from config.json + weights.pt (HuggingFace format)."""
    import json
    from hydra.utils import instantiate

    with open(cfg_path) as f:
        cfg = json.load(f)
    model = instantiate(cfg)
    state_dict = torch.load(wts_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"  Model: {type(model).__name__}, params: {sum(p.numel() for p in model.parameters()):,}")
    return model


@torch.no_grad()
def extract_embeddings(
    checkpoint_path: str,
    h5_path: str,
    output_path: str,
    img_size: int = 224,
    batch_size: int = 2048,
    num_prefetch: int = 12,
    device: str = "cuda:0",
) -> str:
    """Extract LeWM embeddings from all frames in an HDF5 dataset.

    Loads the trained LeWM model, encodes every frame through the frozen
    encoder + projector, and saves the result alongside actions and episode IDs.

    Args:
        checkpoint_path: path to checkpoint (_object.ckpt, dir, or HF repo ID)
        h5_path: path to the HDF5 dataset (e.g., pusht_expert_train.h5)
        output_path: where to save the .npz file
        img_size: image size for normalization
        batch_size: encoding batch size
        num_prefetch: number of batches to prefetch from HDF5 in background
        device: torch device

    Returns:
        output_path
    """
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    # Load the trained model (handles all checkpoint formats)
    model = load_lewm_model(checkpoint_path, device)

    # Access the encoder and projector from the JEPA model
    # The spt.Module wraps: model.model = JEPA(encoder, predictor, ...)
    jepa = model.model if hasattr(model, "model") else model
    encoder = jepa.encoder
    projector = jepa.projector

    # Wrap encoder+projector for multi-GPU if available
    class _EncoderProjector(torch.nn.Module):
        def __init__(self, enc, proj):
            super().__init__()
            self.enc = enc
            self.proj = proj
        def forward(self, x):
            out = self.enc(x, interpolate_pos_encoding=True)
            return self.proj(out.last_hidden_state[:, 0])

    enc_proj = _EncoderProjector(encoder, projector).to(device)
    enc_proj.eval()

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and device != "cpu":
        enc_proj = torch.nn.DataParallel(enc_proj)
        print(f"  Using DataParallel on {n_gpus} GPUs")

    # Load HDF5 data — stream pixels, load metadata
    print(f"Loading HDF5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        h5_keys = list(f.keys())
        print(f"  HDF5 keys: {h5_keys}")

        # Small arrays — load fully
        actions = f["actions"][:] if "actions" in f else f["action"][:]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

        if "ep_idx" in f:
            episode_ids = f["ep_idx"][:].astype(np.int64)
        elif "episode_idx" in f:
            episode_ids = f["episode_idx"][:].astype(np.int64)
        else:
            n_frames = f["pixels"].shape[0]
            episode_ids = np.zeros(n_frames, dtype=np.int64)
            for ei in range(len(ep_len)):
                start = ep_offset[ei]
                end = start + ep_len[ei]
                episode_ids[start:end] = ei

        state = f["state"][:] if "state" in f else None
        proprio = f["proprio"][:] if "proprio" in f else None

        # Pixel metadata — don't load pixels yet
        pix_dset = f["pixels"]
        n_frames = pix_dset.shape[0]
        pix_shape = pix_dset.shape
        is_hwc = (pix_dset.ndim == 4 and pix_dset.shape[-1] == 3)
        is_chw = (pix_dset.ndim == 4 and pix_dset.shape[1] == 3)
        print(f"  Pixel shape: {pix_shape} ({'HWC' if is_hwc else 'CHW' if is_chw else '???'})")

        if not (is_hwc or is_chw):
            raise ValueError(f"Unexpected pixel shape: {pix_shape}")

        # Normalize pixels: uint8 → float32, apply ImageNet-style normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        all_embeddings = []
        n_batches = (n_frames + batch_size - 1) // batch_size

        # Prefetch: read HDF5 batches in background threads while GPU encodes
        def _read_batch(batch_idx):
            s = batch_idx * batch_size
            e = min(s + batch_size, n_frames)
            pix = pix_dset[s:e]
            if is_hwc:
                pix = np.transpose(pix, (0, 3, 1, 2))
            return pix

        print(f"Encoding {n_frames} frames in {n_batches} batches "
              f"(batch_size={batch_size}, prefetch={num_prefetch})...")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=num_prefetch) as pool:
            # Submit first N batches
            futures = {}
            for i in range(min(num_prefetch, n_batches)):
                futures[i] = pool.submit(_read_batch, i)

            for b in range(n_batches):
                if (b + 1) % 20 == 0 or b == n_batches - 1:
                    elapsed = time.time() - t0
                    fps = (b + 1) * batch_size / max(elapsed, 1e-6)
                    eta = (n_batches - b - 1) * elapsed / max(b + 1, 1)
                    print(f"  batch {b + 1}/{n_batches}  ({fps:.0f} frames/s, ETA {eta:.0f}s)")

                # Get prefetched pixels
                batch_pix = futures[b].result()

                # Submit next batch for prefetch
                next_b = b + num_prefetch
                if next_b < n_batches:
                    futures[next_b] = pool.submit(_read_batch, next_b)

                # Clean up used future
                del futures[b]

                batch_pix = torch.from_numpy(batch_pix).float().to(device) / 255.0
                batch_pix = (batch_pix - mean) / std

                emb = enc_proj(batch_pix)
                all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_dict = dict(
        embeddings=embeddings,
        actions=actions.astype(np.float32),
        episode_ids=episode_ids,
        embed_dim=embeddings.shape[1],
    )
    if state is not None:
        save_dict["state"] = state.astype(np.float32)
    if proprio is not None:
        save_dict["proprio"] = proprio.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved: {output_path}")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  actions:    {actions.shape}")
    if state is not None:
        print(f"  state:      {state.shape}")
    if proprio is not None:
        print(f"  proprio:    {proprio.shape}")
    print(f"  episodes:   {len(ep_len)}")
    return output_path
