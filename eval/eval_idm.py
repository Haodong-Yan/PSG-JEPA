#!/usr/bin/env python3
"""Evaluate GC-IDM vs CEM planning on LeWM.

Usage:
    python eval_idm.py --dataset tworoom --idm ./checkpoints/tworoom_gcidm.pt
    python eval_idm.py --dataset tworoom --cem-only
    python eval_idm.py --dataset tworoom --idm ./checkpoints/tworoom_gcidm.pt --compare
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import stable_worldmodel as swm
import stable_pretraining as spt
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors

from idm.dataset import load_lewm_model
from idm.model import IDMConfig
from train_idm import load_idm


# ---------------------------------------------------------------------------
# Config per dataset
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    env: str
    dataset_name: str
    cache_keys: list[str]
    img_size: int
    embed_dim: int
    action_block: int
    raw_action_dim: int
    callables: list[dict]
    world_kwargs: dict | None = None


DATASET_CONFIGS = {
    "pusht": EvalConfig(
        env="swm/PushT-v1",
        dataset_name="pusht_expert_train",
        cache_keys=["action", "proprio", "state"],
        img_size=224, embed_dim=192, action_block=1, raw_action_dim=2,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    ),
    "tworoom": EvalConfig(
        env="swm/TwoRoom-v1",
        dataset_name="tworoom",
        cache_keys=["action", "proprio"],
        img_size=224, embed_dim=192, action_block=1, raw_action_dim=2,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
    ),
    "cube": EvalConfig(
        env="swm/OGBCube-v0",
        dataset_name="ogbench/cube_single_expert",
        cache_keys=["action"],
        img_size=224, embed_dim=192, action_block=1, raw_action_dim=5,
        callables=[
            {"method": "set_state", "args": {
                "qpos": {"value": "qpos"}, "qvel": {"value": "qvel"},
            }},
            {"method": "set_target_pos", "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            }},
        ],
        world_kwargs={"env_type": "single", "ob_type": "states", "multiview": False,
                       "width": 224, "height": 224, "visualize_info": False,
                       "terminate_at_goal": True},
    ),
    "reacher": EvalConfig(
        env="swm/ReacherDMControl-v0",
        dataset_name="dmc/reacher_random",
        cache_keys=["action"],
        img_size=224, embed_dim=192, action_block=1, raw_action_dim=2,
        callables=[
            {"method": "set_state", "args": {
                "qpos": {"value": "qpos"}, "qvel": {"value": "qvel"},
            }},
            {"method": "set_target_qpos", "args": {
                "target_qpos": {"value": "goal_qpos"},
            }},
        ],
        world_kwargs={"task": "qpos_match"},
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_img_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def build_normalizers(dataset, keys):
    process = {}
    for col in keys:
        if col == "pixels":
            continue
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler = preprocessing.StandardScaler().fit(col_data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def sample_eval_episodes(dataset, num_eval, goal_offset, seed=42,
                         train_split=1.0, split_seed=42):
    """Sample eval episodes. If train_split < 1.0, restrict to held-out episodes only."""
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    ep_indices = np.unique(episode_idx)

    episode_len = np.array([
        np.max(step_idx[episode_idx == ep]) + 1 for ep in ep_indices
    ])
    max_start = episode_len - goal_offset - 1
    max_start_dict = {ep: mx for ep, mx in zip(ep_indices, max_start)}
    max_start_per_row = np.array([max_start_dict[ep] for ep in episode_idx])
    valid_mask = step_idx <= max_start_per_row

    # If train_split < 1.0, restrict to held-out episodes
    if train_split < 1.0:
        n_holdout = max(1, round(len(ep_indices) * (1.0 - train_split)))
        rng_split = np.random.default_rng(split_seed)
        holdout_eps = rng_split.choice(ep_indices, size=n_holdout, replace=False)
        valid_mask = valid_mask & np.isin(episode_idx, holdout_eps)
        print(f"  Eval restricted to {len(holdout_eps)} held-out episodes "
              f"(split_seed={split_seed}, train_split={train_split})")

    valid_indices = np.nonzero(valid_mask)[0]

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No valid eval starts (goal_offset={goal_offset}, "
            f"train_split={train_split})")

    rng = np.random.default_rng(seed)
    n = min(num_eval, len(valid_indices) - 1)
    if n < num_eval:
        print(f"  WARNING: reduced num_eval {num_eval} -> {n} "
              f"(only {len(valid_indices)} valid starts)")
    sampled = rng.choice(len(valid_indices) - 1, size=n, replace=False)
    sampled = np.sort(valid_indices[sampled])

    eval_episodes = dataset.get_row_data(sampled)[col_name]
    eval_start_idx = dataset.get_row_data(sampled)["step_idx"]
    return eval_episodes, eval_start_idx


# ---------------------------------------------------------------------------
# GoalConditionedPolicy
# ---------------------------------------------------------------------------

class GoalConditionedPolicy:
    """One forward pass per step: encode obs + goal → IDM → action."""

    def __init__(self, jepa_model, idm, eval_budget=50,
                 process=None, transform=None,
                 device=torch.device("cpu"), cache_goal_encoding=True):
        self.jepa = jepa_model
        self.idm = idm
        self.eval_budget = eval_budget
        self.process = process or {}
        self.transform = transform or {}
        self.device = device
        self._step_count = 0
        self._plan_times: list[float] = []
        self.cache_goal_encoding = cache_goal_encoding
        self._cached_goal_input = None
        self._cached_z_goal = None

    def set_env(self, env):
        self.env = env

    def _prepare_info(self, info_dict):
        out = {}
        for k, v in list(info_dict.items()):
            is_numpy = isinstance(v, (np.ndarray, np.generic))

            if k in self.process and is_numpy:
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            if k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith("pixels") or k.startswith("goal"):
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack([self.transform[k](tv_tensors.Image(x)) for x in v])
                is_numpy = isinstance(v, (np.ndarray, np.generic))
                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in "USO":
                v = torch.from_numpy(v)
            out[k] = v
        return out

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        t0 = time.perf_counter()
        info_dict = self._prepare_info(info_dict)

        pixels = info_dict["pixels"][:, -1].to(self.device)
        goal = info_dict["goal"][:, -1].to(self.device)

        enc_out = self.jepa.encoder(pixels, interpolate_pos_encoding=True)
        z_current = self.jepa.projector(enc_out.last_hidden_state[:, 0])

        if (self.cache_goal_encoding
                and self._cached_goal_input is not None
                and self._cached_goal_input.shape == goal.shape
                and torch.equal(self._cached_goal_input, goal)):
            z_goal = self._cached_z_goal
        else:
            enc_out_g = self.jepa.encoder(goal, interpolate_pos_encoding=True)
            z_goal = self.jepa.projector(enc_out_g.last_hidden_state[:, 0])
            if self.cache_goal_encoding:
                self._cached_goal_input = goal
                self._cached_z_goal = z_goal

        remaining = min(self.eval_budget - self._step_count, self.idm.max_horizon)
        steps_remaining = torch.full(
            (z_current.shape[0],), remaining, device=self.device, dtype=torch.long,
        )

        action = self.idm(z_current, z_goal, steps_remaining)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._plan_times.append(time.perf_counter() - t0)
        self._step_count += 1

        return action.cpu().reshape(*self.env.action_space.shape).numpy()

    @property
    def avg_plan_time_ms(self):
        if not self._plan_times:
            return 0.0
        return 1000 * sum(self._plan_times) / len(self._plan_times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GC-IDM vs CEM")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--idm", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cem-only", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--no-goal-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=1.0,
                        help="If < 1.0, eval only on held-out episodes (match training split)")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-name-override", default=None)
    args = parser.parse_args()

    dcfg = DATASET_CONFIGS[args.dataset]
    if args.dataset_name_override is not None:
        import dataclasses
        dcfg = dataclasses.replace(dcfg, dataset_name=args.dataset_name_override)
    device = torch.device(args.device)
    data_dir = os.environ.get("STABLEWM_HOME", "./stable-wm-data")
    ckpt_dir = os.path.join(data_dir, "checkpoints", args.dataset, "lewm")
    ckpt_path = args.checkpoint or ckpt_dir

    # Load dataset + transforms
    dataset = swm.data.HDF5Dataset(dcfg.dataset_name, keys_to_cache=dcfg.cache_keys)
    img_tf = get_img_transform(dcfg.img_size)
    transform = {"pixels": img_tf, "goal": img_tf}
    process = build_normalizers(dataset, dcfg.cache_keys)

    eval_episodes, eval_start_idx = sample_eval_episodes(
        dataset, args.num_eval, args.goal_offset, args.seed,
        train_split=args.train_split, split_seed=args.split_seed,
    )

    results = {}

    # --- CEM ---
    if args.cem_only or args.compare:
        print("\n" + "=" * 60)
        print("  CEM Planning Evaluation")
        print("=" * 60)

        ckpt_obj = os.path.join(ckpt_path, "lewm_object.ckpt")
        if not os.path.exists(ckpt_obj) and os.path.exists(os.path.join(ckpt_path, "config.json")):
            _model = load_lewm_model(ckpt_path, "cpu")
            torch.save(_model, ckpt_obj)
            del _model

        cost_model = swm.policy.AutoCostModel(f"{args.dataset}/lewm")
        cost_model = cost_model.to(device).eval()
        cost_model.requires_grad_(False)
        cost_model.interpolate_pos_encoding = True

        config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
        solver = swm.solver.CEMSolver(model=cost_model, device=device, seed=args.seed)
        cem_policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform,
        )

        world = swm.World(
            env_name=dcfg.env, num_envs=args.num_eval,
            image_shape=(dcfg.img_size, dcfg.img_size),
            max_episode_steps=2 * max(args.eval_budget, args.goal_offset) + 5,
            **(dcfg.world_kwargs or {}),
        )
        world.set_policy(cem_policy)

        # Warmup
        _dummy = torch.randn(min(args.num_eval, 8), 3, dcfg.img_size, dcfg.img_size, device=device)
        with torch.no_grad():
            for _ in range(3):
                cost_model.encoder(_dummy, interpolate_pos_encoding=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        del _dummy

        _ep_arr = np.array(eval_episodes.tolist())
        _start_arr = np.array(eval_start_idx.tolist())
        _ = dataset.load_chunk(_ep_arr, _start_arr, _start_arr + args.goal_offset + 1)
        del _

        cem_plan_times: list[float] = []
        _orig = cem_policy.get_action
        def _timed(info_dict, **kw):
            if device.type == "cuda": torch.cuda.synchronize()
            _t = time.perf_counter()
            _r = _orig(info_dict, **kw)
            if device.type == "cuda": torch.cuda.synchronize()
            cem_plan_times.append(time.perf_counter() - _t)
            return _r
        cem_policy.get_action = _timed

        t0 = time.time()
        cem_metrics = world.evaluate(
            dataset=dataset, start_steps=eval_start_idx.tolist(),
            goal_offset=args.goal_offset, eval_budget=args.eval_budget,
            episodes_idx=eval_episodes.tolist(), callables=dcfg.callables,
        )
        cem_time = time.time() - t0
        cem_plan_ms = 1000 * sum(cem_plan_times) / len(cem_plan_times) if cem_plan_times else 0.0

        results["cem"] = cem_metrics
        results["cem_ms"] = 1000 * cem_time / args.num_eval
        results["cem_plan_ms"] = cem_plan_ms
        print(f"CEM metrics: {cem_metrics}")
        print(f"CEM wall-clock: {cem_time:.1f}s total, {results['cem_ms']:.0f} ms/episode")
        print(f"CEM pure-compute: {cem_plan_ms:.1f} ms/plan-call ({len(cem_plan_times)} calls)")

    # --- GC-IDM ---
    if not args.cem_only:
        assert args.idm, "--idm required"
        print("\n" + "=" * 60)
        print("  GC-IDM Evaluation")
        print("=" * 60)

        model = load_lewm_model(ckpt_path, str(device))
        jepa = model.model if hasattr(model, "model") else model
        jepa.eval()
        jepa.requires_grad_(False)

        idm = load_idm(args.idm, str(device))

        # Speed benchmark
        z_test = torch.randn(1, dcfg.embed_dim, device=device)
        for _ in range(10):
            idm(z_test, z_test, torch.tensor([25], device=device))
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(500):
            idm(z_test, z_test, torch.tensor([25], device=device))
        if device.type == "cuda": torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / 500
        print(f"GC-IDM speed: {ms:.2f} ms/forward ({1000/ms:.0f} Hz)")

        idm_policy = GoalConditionedPolicy(
            jepa_model=jepa, idm=idm, eval_budget=args.eval_budget,
            process=process, transform=transform, device=device,
            cache_goal_encoding=not args.no_goal_cache,
        )

        world = swm.World(
            env_name=dcfg.env, num_envs=args.num_eval,
            image_shape=(dcfg.img_size, dcfg.img_size),
            max_episode_steps=2 * max(args.eval_budget, args.goal_offset) + 5,
            **(dcfg.world_kwargs or {}),
        )
        world.set_policy(idm_policy)

        # Warmup
        _dummy = torch.randn(min(args.num_eval, 8), 3, dcfg.img_size, dcfg.img_size, device=device)
        with torch.no_grad():
            for _ in range(3):
                jepa.encoder(_dummy, interpolate_pos_encoding=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        del _dummy

        _ep_arr = np.array(eval_episodes.tolist())
        _start_arr = np.array(eval_start_idx.tolist())
        _ = dataset.load_chunk(_ep_arr, _start_arr, _start_arr + args.goal_offset + 1)
        del _

        t0 = time.time()
        idm_metrics = world.evaluate(
            dataset=dataset, start_steps=eval_start_idx.tolist(),
            goal_offset=args.goal_offset, eval_budget=args.eval_budget,
            episodes_idx=eval_episodes.tolist(), callables=dcfg.callables,
        )
        idm_time = time.time() - t0

        results["idm"] = idm_metrics
        results["idm_ms"] = 1000 * idm_time / args.num_eval
        results["idm_plan_ms"] = idm_policy.avg_plan_time_ms
        print(f"IDM metrics: {idm_metrics}")
        print(f"IDM time: {idm_time:.1f}s total, {results['idm_ms']:.0f} ms/episode")
        print(f"IDM avg plan time: {idm_policy.avg_plan_time_ms:.2f} ms")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"{'Method':<20} {'Success Rate':>15} {'ms/episode':>12} {'ms/plan':>10}")
    print("-" * 60)

    if "cem" in results:
        sr = results["cem"].get("success_rate", "?")
        print(f"{'CEM':<20} {str(sr)+'%':>15} {results['cem_ms']:>12.0f} {results['cem_plan_ms']:>10.0f}")

    if "idm" in results:
        sr = results["idm"].get("success_rate", "?")
        print(f"{'GC-IDM':<20} {str(sr)+'%':>15} {results['idm_ms']:>12.0f} {results['idm_plan_ms']:>10.1f}")

    if "cem" in results and "idm" in results:
        speedup = results["cem_plan_ms"] / max(results["idm_plan_ms"], 0.1)
        delta = results["idm"].get("success_rate", 0) - results["cem"].get("success_rate", 0)
        print(f"\nSpeedup: {speedup:.0f}x")
        print(f"Success rate delta: {delta:+.1f}%")


if __name__ == "__main__":
    main()
