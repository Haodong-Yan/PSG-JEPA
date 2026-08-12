"""Train PSG-JEPA on OGBench-Cube.

PSG-JEPA = LeWM's JEPA world model (forward prediction + SIGReg) + physical state grounding:
  static grounding   H_s : z_t          -> proprioceptive state s_t
  dynamic grounding  H_d : (z_t, z_{t+k}) -> multi-horizon joint-angle change  (+ optional velocity)
Grounding heads are training-only and discarded at inference (see psgjepa/grounding.py).

Usage:
  python train.py data=ogb_cm subdir=psgjepa_cube
  # disable the optional velocity term (envs without a velocity signal):
  python train.py data=ogb_cm loss.grounding.use_velocity=false
"""
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from psgjepa.model import CrossModalViTEncoder, CrossModalJEPA_v2, psg_forward
from psgjepa.grounding import PSGGroundingHeads
from psgjepa.module import ARPredictor, Embedder, MLP, SIGReg
from psgjepa.utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack, RenameKey


@hydra.main(version_base=None, config_path="configs", config_name="psgjepa")
def run(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    pixels_key = cfg.get("pixels_key", "pixels")
    transforms = []
    if pixels_key != "pixels":
        transforms.append(RenameKey(pixels_key, "pixels"))
    transforms.append(get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
        for col in dataset.column_names:
            if col not in cfg.data.dataset.keys_to_load and col != "pixels":
                if hasattr(cfg.wm, f"{col}_dim"):
                    continue
                transforms.append(get_column_normalizer(dataset, col, col))
                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    # --- world model: shared cross-modal ViT encoder + autoregressive predictor ---
    raw_vit = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False,
    )
    hidden_dim = raw_vit.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    use_state = cfg.wm.get("use_state", True)
    proprio_dim = cfg.wm.get("proprio_dim", 2) if use_state else 0

    encoder = CrossModalViTEncoder(
        vit_model=raw_vit, proprio_dim=proprio_dim, patch_size=cfg.patch_size,
        state_inject_mode=cfg.wm.get("state_inject_mode", "patch"),
    )
    predictor = ARPredictor(
        num_frames=cfg.wm.history_size, input_dim=embed_dim,
        hidden_dim=hidden_dim, output_dim=hidden_dim, **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    # --- physical state grounding heads (training-only) ---
    grounding = None
    gcfg = cfg.loss.get("grounding", {})
    gw = float(gcfg.get("weight", 0.0))
    if gw > 0:
        use_vel = bool(gcfg.get("use_velocity", False))
        grounding = PSGGroundingHeads(
            embed_dim=embed_dim,
            state_dim=len(gcfg.state_idx),
            joint_dim=len(gcfg.joint_idx),
            vel_dim=len(gcfg.vel_idx) if use_vel else None,
            use_velocity=use_vel,
            hidden=gcfg.get("hidden_dim", 256),
        )
        print(f"[grounding] weight={gw} state_dim={len(gcfg.state_idx)} "
              f"joint_dim={len(gcfg.joint_idx)} use_velocity={use_vel} "
              f"params={sum(p.numel() for p in grounding.parameters())/1e6:.2f}M")

    world_model = CrossModalJEPA_v2(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj, use_state=use_state, grounding=grounding,
    )

    optimizers = {"model_opt": {
        "modules": "model", "optimizer": dict(cfg.optimizer),
        "scheduler": {"type": "LinearWarmupCosineAnnealingLR"}, "interval": "epoch",
    }}
    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(psg_forward, cfg=cfg),
        optim=optimizers,
    )

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    cb = ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1)
    trainer = pl.Trainer(**cfg.trainer, callbacks=[cb], num_sanity_val_steps=1,
                         logger=logger, enable_checkpointing=True)
    manager = spt.Manager(
        trainer=trainer, module=module, data=data_module, seed=cfg.seed,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
