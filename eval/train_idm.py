#!/usr/bin/env python3
"""Train a Goal-Conditioned IDM on LeWM embeddings.

Usage:
    python train_idm.py extract --checkpoint <ckpt> --h5 <dataset.h5> --output embeddings.npz
    python train_idm.py train --embeddings embeddings.npz --output idm.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from idm.model import GoalConditionedIDM, IDMConfig
from idm.dataset import EmbeddingTripleDataset, extract_embeddings


def train_idm(
    embeddings_path: str,
    output_path: str,
    embed_dim: int = 192,
    action_dim: int = 2,
    frameskip: int = 1,
    hidden_dim: int = 512,
    n_layers: int = 3,
    noise_sigma: float = 0.0,
    noise_schedule: str = "fixed",
    max_goal_horizon: int = 50,
    train_split: float = 1.0,
    split_seed: int = 42,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 8192,
    epochs: int = 200,
    val_split: float = 0.1,
    seed: int = 42,
    device: str = "cuda:0",
    save_epochs: list | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = IDMConfig(
        embed_dim=embed_dim, action_dim=action_dim, frameskip=frameskip,
        hidden_dim=hidden_dim, n_layers=n_layers,
        noise_sigma=noise_sigma, noise_schedule=noise_schedule,
        max_horizon=max_goal_horizon,
    )

    dataset = EmbeddingTripleDataset(
        embeddings_path,
        frameskip=frameskip, max_goal_horizon=max_goal_horizon,
        train_split=train_split, split_seed=split_seed,
    )
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=12, pin_memory=True,
    )

    print(f"Dataset: {len(dataset)} pairs ({n_train} train, {n_val} val)")
    print(f"Config: {cfg}")

    model = GoalConditionedIDM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"IDM parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mse_per_dim": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            z_t = batch["z_t"].to(device)
            z_goal = batch["z_goal"].to(device)
            steps = batch["steps_remaining"].to(device)
            action = batch["action"].to(device)

            pred = model(z_t, z_goal, steps)
            loss = nn.functional.mse_loss(pred, action)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                z_t = batch["z_t"].to(device)
                z_goal = batch["z_goal"].to(device)
                steps = batch["steps_remaining"].to(device)
                action = batch["action"].to(device)

                pred = model(z_t, z_goal, steps)
                loss = nn.functional.mse_loss(pred, action)
                val_losses.append(loss.item())
                all_preds.append(pred.cpu())
                all_targets.append(action.cpu())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        per_dim_mse = ((preds_cat - targets_cat) ** 2).mean(dim=0).tolist()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse_per_dim"].append(per_dim_mse)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "mode": "goal_conditioned",
                "epoch": epoch,
                "val_loss": val_loss,
                "train_split": train_split,
                "split_seed": split_seed,
                "held_out_episodes": dataset.held_out_episodes,
            }, output_path)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>4d}/{epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"best={best_val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
        # Save the weights AT the listed epochs (--output above keeps only the best-val epoch).
        if save_epochs is not None and (epoch + 1) in save_epochs:
            ep_out = output_path.replace(".pt", f"_ep{epoch+1}.pt")
            torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__, "mode": "goal_conditioned",
                        "epoch": epoch, "val_loss": val_loss, "train_split": train_split,
                        "split_seed": split_seed, "held_out_episodes": dataset.held_out_episodes}, ep_out)
            print(f"  [save] {ep_out}")

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved: {output_path}")

    with open(output_path.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f)

    return {"best_val_loss": best_val_loss, "n_params": n_params, "epochs": epochs}


def load_idm(idm_path: str, device: str = "cuda:0") -> torch.nn.Module:
    """Load a trained IDM from checkpoint."""
    data = torch.load(idm_path, map_location=device, weights_only=False)
    cfg = IDMConfig(**data["config"])
    model = GoalConditionedIDM(cfg)
    model.load_state_dict(data["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="GC-IDM training")
    subparsers = parser.add_subparsers(dest="command")

    p_ext = subparsers.add_parser("extract", help="Extract embeddings from trained LeWM")
    p_ext.add_argument("--checkpoint", required=True)
    p_ext.add_argument("--h5", required=True)
    p_ext.add_argument("--output", required=True)
    p_ext.add_argument("--img-size", type=int, default=224)
    p_ext.add_argument("--batch-size", type=int, default=2048)
    p_ext.add_argument("--num-prefetch", type=int, default=12,
                       help="HDF5 read-ahead batches (background threads)")
    p_ext.add_argument("--device", default="cuda:0")

    p_train = subparsers.add_parser("train", help="Train GC-IDM")
    p_train.add_argument("--embeddings", required=True)
    p_train.add_argument("--output", required=True)
    p_train.add_argument("--embed-dim", type=int, default=192)
    p_train.add_argument("--action-dim", type=int, default=2)
    p_train.add_argument("--frameskip", type=int, default=1)
    p_train.add_argument("--hidden-dim", type=int, default=512)
    p_train.add_argument("--n-layers", type=int, default=3)
    p_train.add_argument("--noise-sigma", type=float, default=0.0)
    p_train.add_argument("--noise-schedule", default="fixed")
    p_train.add_argument("--max-goal-horizon", type=int, default=50)
    p_train.add_argument("--train-split", type=float, default=1.0,
                         help="Episode-level train split (0.9 = match LeWM)")
    p_train.add_argument("--split-seed", type=int, default=42,
                         help="Seed for episode split (42 = match paper)")
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--batch-size", type=int, default=8192)
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--save-epochs", type=str, default=None,
                         help="comma-separated epoch numbers to save ckpts at")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    if args.command == "extract":
        extract_embeddings(
            checkpoint_path=args.checkpoint, h5_path=args.h5,
            output_path=args.output, img_size=args.img_size,
            batch_size=args.batch_size, num_prefetch=args.num_prefetch,
            device=args.device,
        )
    elif args.command == "train":
        train_idm(
            embeddings_path=args.embeddings, output_path=args.output,
            embed_dim=args.embed_dim, action_dim=args.action_dim,
            frameskip=args.frameskip, hidden_dim=args.hidden_dim,
            n_layers=args.n_layers, noise_sigma=args.noise_sigma,
            noise_schedule=args.noise_schedule,
            max_goal_horizon=args.max_goal_horizon,
            train_split=args.train_split, split_seed=args.split_seed,
            lr=args.lr, batch_size=args.batch_size,
            epochs=args.epochs, seed=args.seed, device=args.device,
            save_epochs=([int(x) for x in args.save_epochs.split(",")] if args.save_epochs else None),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
