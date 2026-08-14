"""Convert LIBERO-Goal demos into the single HDF5 file the encoder pretraining reads.

Output schema (matches stable_worldmodel.data.HDF5Dataset):
  pixels    : (N, H, W, 3) uint8   agentview, every atomic frame
  action    : (N, 7)       float32 raw LIBERO actions
  ep_len    : (E,)         int64   per-episode atomic length
  ep_offset : (E,)         int64   cumulative start offset

Each frame is preprocessed exactly as the action head will see it -- rotate 180, center-crop 0.9,
resize to --img-size (bilinear) -- so the pretrained encoder and the downstream head share one
view geometry; both then apply the ImageNet normalization and resize to 224.

`--with-state` additionally stores the robot-body state, which is the grounding target. LIBERO
demos expose robot proprioception only, with no velocity field, and none is synthesized, so this
is a position-only state:

  observation = proprio = [joint_states(7), ee_pos(3), ee_ori(3), gripper_states(2)] = 15-d

Both `observation` (the grounding target) and `proprio` (an optional encoder input, unused when
wm.use_state=false) are written.

Usage:
  python libero/convert_libero_to_swm.py --with-state --out-name libero_goal_cm_state
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


# LIBERO-Goal demos, as laid out by a clone of Lifelong-Robot-Learning/LIBERO.
DEFAULT_GOAL_DIR = (Path(__file__).resolve().parents[1]
                    / "assets" / "benchmarks" / "LIBERO" / "libero" / "datasets" / "libero_goal")


def get_cache_dir() -> Path:
    return Path(os.getenv("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel")))


def prep_image(img: np.ndarray, img_size: int, rotate180: bool, crop_frac: float) -> np.ndarray:
    if rotate180:
        img = img[::-1, ::-1]
    if crop_frac and crop_frac < 1.0:
        h, w = img.shape[:2]
        ch, cw = int(h * crop_frac), int(w * crop_frac)
        top, left = (h - ch) // 2, (w - cw) // 2
        img = img[top:top + ch, left:left + cw]
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = np.asarray(
            Image.fromarray(np.ascontiguousarray(img)).resize((img_size, img_size), Image.BILINEAR)
        )
    return np.ascontiguousarray(img.astype(np.uint8, copy=False))


def demo_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-goal-dir", type=Path, default=DEFAULT_GOAL_DIR,
                    help="directory holding the LIBERO-Goal demo HDF5 files")
    ap.add_argument("--out-name", default="libero_goal_cm")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--rotate180", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crop-frac", type=float, default=0.9)
    ap.add_argument("--image-key", default="agentview_rgb")
    ap.add_argument("--with-state", action="store_true",
                    help="also store observation+proprio = robot-body-only 15-d position state")
    args = ap.parse_args()

    # robot-body-only proprio: joint(7)+ee_pos(3)+ee_ori(3)+gripper(2)=15, NO object/vel
    STATE_KEYS = [("joint_states", 7), ("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)]
    STATE_DIM = sum(d for _, d in STATE_KEYS)

    cache_dir = args.cache_dir or get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{args.out_name}.h5"

    files = sorted(args.libero_goal_dir.glob("*_demo.hdf5"))
    if not files:
        raise FileNotFoundError(f"No *_demo.hdf5 under {args.libero_goal_dir}")
    print(f"Found {len(files)} task files. Writing -> {out_path}")

    S = args.img_size
    ep_len = []
    total = 0
    if out_path.exists():
        out_path.unlink()
    with h5py.File(out_path, "w") as out:
        px = out.create_dataset(
            "pixels", shape=(0, S, S, 3), maxshape=(None, S, S, 3),
            dtype="uint8", chunks=(64, S, S, 3), compression="gzip", compression_opts=4,
        )
        ac = out.create_dataset(
            "action", shape=(0, 7), maxshape=(None, 7), dtype="float32", chunks=(4096, 7),
        )
        obs_ds = pr_ds = None
        if args.with_state:
            obs_ds = out.create_dataset(
                "observation", shape=(0, STATE_DIM), maxshape=(None, STATE_DIM),
                dtype="float32", chunks=(4096, STATE_DIM),
            )
            pr_ds = out.create_dataset(
                "proprio", shape=(0, STATE_DIM), maxshape=(None, STATE_DIM),
                dtype="float32", chunks=(4096, STATE_DIM),
            )
        for fi, fpath in enumerate(files):
            with h5py.File(fpath, "r") as h:
                demos = sorted(h["data"].keys(), key=demo_sort_key)
                for dn in demos:
                    d = h["data"][dn]
                    raw_px = np.asarray(d["obs"][args.image_key])  # (T,128,128,3)
                    act = np.asarray(d["actions"], dtype=np.float32)  # (T,7)
                    T = act.shape[0]
                    frames = np.stack([prep_image(raw_px[i], S, args.rotate180, args.crop_frac)
                                       for i in range(T)], axis=0)
                    px.resize(total + T, axis=0); px[total:total + T] = frames
                    ac.resize(total + T, axis=0); ac[total:total + T] = act
                    if args.with_state:
                        state = np.concatenate(
                            [np.asarray(d["obs"][k], dtype=np.float32)[:, :dim] for k, dim in STATE_KEYS],
                            axis=1,
                        )  # (T, STATE_DIM)
                        assert state.shape == (T, STATE_DIM), f"{state.shape} != {(T, STATE_DIM)}"
                        obs_ds.resize(total + T, axis=0); obs_ds[total:total + T] = state
                        pr_ds.resize(total + T, axis=0); pr_ds[total:total + T] = state
                    ep_len.append(T)
                    total += T
            print(f"  [{fi+1}/{len(files)}] {fpath.name}: {len(demos)} demos, running total {total} frames")
        ep_len = np.asarray(ep_len, dtype=np.int64)
        ep_offset = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
        out.create_dataset("ep_len", data=ep_len)
        out.create_dataset("ep_offset", data=ep_offset)

    print(f"DONE: {out_path}")
    print(f"  episodes={len(ep_len)}  total_frames={total}  pixels=({total},{S},{S},3)uint8  action=({total},7)")
    if args.with_state:
        print(f"  observation=proprio=({total},{STATE_DIM}) robot-body-only [joint7+ee_pos3+ee_ori3+gripper2]")


if __name__ == "__main__":
    main()
