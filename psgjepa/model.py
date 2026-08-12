"""PSG-JEPA world model for OGBench-Cube: shared cross-modal ViT encoder,
autoregressive predictor (JEPA forward prediction + SIGReg), and physical state grounding.
Grounding heads (psgjepa/grounding.py) are training-only and discarded at inference.
"""
"""Cross-modal JEPA v2: state tokens injected into ViT as extra patches.

Each proprio dim → constant 14×14×3 block → same patch_embed Conv2d as video
→ state tokens live alongside video tokens inside one ViT forward pass.
Self-attention naturally fuses video ↔ state.

No extra encoder, no extra Linear — state goes through the exact same
patch_embed + transformer as video patches.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class CrossModalViTEncoder(nn.Module):
    """Wraps a HuggingFace ViT to jointly encode video patches + state tokens.

    state_inject_mode:
      - "patch":  each proprio scalar → constant (3, ps, ps) block → same
                  patch_embed Conv2d as video. Zero extra params for encoding.
      - "linear": nn.Linear(1, D) per-dim embedding (Cosmos/ViVa style).
                  Adds proprio_dim * D params, but each dim gets an independent
                  direction in embedding space.
      - "linear_shared": nn.Linear(proprio_dim, D) single projection.
                  Lightest learned variant.

    Extra learnable params (all modes):
      - modality_emb_state (1, 1, D)
      - state_pos_emb (1, max_proprio, D)
    """

    def __init__(self, vit_model, proprio_dim: int, patch_size: int = 14,
                 state_inject_mode: str = "patch"):
        super().__init__()
        self.vit = vit_model
        self.proprio_dim = proprio_dim
        self.patch_size = patch_size
        self.state_inject_mode = state_inject_mode
        hidden_dim = vit_model.config.hidden_size

        self.modality_emb_state = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.state_pos_emb = nn.Parameter(
            torch.randn(1, proprio_dim, hidden_dim) * 0.02
        )

        if state_inject_mode == "linear":
            self.state_proj_per_dim = nn.Linear(1, hidden_dim)
        elif state_inject_mode == "linear_shared":
            self.state_proj_shared = nn.Linear(proprio_dim, hidden_dim)
        # "repeat_tile" needs no extra params

    @property
    def config(self):
        return self.vit.config

    def _make_state_blocks(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, proprio_dim) → (B, proprio_dim, 3, ps, ps)"""
        B, D = state.shape
        ps = self.patch_size
        blocks = state[:, :, None, None, None].expand(B, D, 3, ps, ps)
        return blocks.contiguous()

    def _embed_state_tokens(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, proprio_dim) → (B, N_tokens, hidden_dim)

        N_tokens depends on mode:
          patch / linear      → proprio_dim tokens (one per dim)
          repeat_tile         → 1 token  (all dims tiled into one vector)
          linear_shared       → 1 token  (single projection)
        """
        B, D = state.shape
        hidden = self.vit.config.hidden_size

        if self.state_inject_mode == "patch":
            blocks = self._make_state_blocks(state)
            patch_proj = self.vit.embeddings.patch_embeddings.projection
            flat = blocks.reshape(B * D, 3, self.patch_size, self.patch_size)
            tokens = patch_proj(flat).flatten(1)              # (B*D, hidden)
            return tokens.reshape(B, D, -1)

        elif self.state_inject_mode == "linear":
            s = state.unsqueeze(-1)                           # (B, D, 1)
            return self.state_proj_per_dim(s)                 # (B, D, hidden)

        elif self.state_inject_mode == "repeat_tile":
            # Cosmos/ViVa style: flatten + repeat-tile to hidden_dim, no learnable params
            n_rep = (hidden + D - 1) // D
            tiled = state.repeat(1, n_rep)[:, :hidden]        # (B, hidden)
            return tiled.unsqueeze(1)                          # (B, 1, hidden)

        elif self.state_inject_mode == "linear_shared":
            return self.state_proj_shared(state).unsqueeze(1)  # (B, 1, hidden)

        else:
            raise ValueError(f"Unknown state_inject_mode: {self.state_inject_mode}")

    def forward(self, pixels, proprio=None, interpolate_pos_encoding=True):
        """
        pixels: (B, 3, H, W)
        proprio: (B, proprio_dim) or None

        Returns: SimpleNamespace with .last_hidden_state
          - If proprio is None: (B, 1+N_vid, D)
          - If proprio given:   (B, 1+N_vid+N_state, D)
            where N_vid=256, N_state=proprio_dim
        """
        B = pixels.shape[0]
        embeddings = self.vit.embeddings
        patch_proj = embeddings.patch_embeddings.projection

        # --- Video tokens ---
        vid_out = patch_proj(pixels)                          # (B, D, H', W')
        vid_tokens = vid_out.flatten(2).transpose(1, 2)      # (B, N_vid, D)
        N_vid = vid_tokens.shape[1]

        # --- CLS token ---
        cls_tokens = embeddings.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # --- Position encoding for [CLS + video] ---
        pos_emb = embeddings.position_embeddings              # (1, 1+N_vid, D)
        if interpolate_pos_encoding and pos_emb.shape[1] != 1 + N_vid:
            cls_pos = pos_emb[:, :1]
            vid_pos = pos_emb[:, 1:]
            n_orig = int(math.sqrt(vid_pos.shape[1]))
            n_new = int(math.sqrt(N_vid))
            vid_pos = vid_pos.reshape(1, n_orig, n_orig, -1).permute(0, 3, 1, 2)
            vid_pos = F.interpolate(vid_pos, size=(n_new, n_new), mode="bilinear")
            vid_pos = vid_pos.permute(0, 2, 3, 1).reshape(1, -1, vid_pos.shape[1])
            pos_emb = torch.cat([cls_pos, vid_pos], dim=1)

        vid_with_cls = torch.cat([cls_tokens, vid_tokens], dim=1)
        vid_with_cls = vid_with_cls + pos_emb[:, :1 + N_vid]
        vid_with_cls = embeddings.dropout(vid_with_cls)

        state_inject_mode = getattr(self, 'state_inject_mode', 'patch')
        if proprio is None or state_inject_mode == "late_concat":
            hidden = self.vit.encoder(vid_with_cls).last_hidden_state
            hidden = self.vit.layernorm(hidden)
            if state_inject_mode == "late_concat":
                # Late fusion: store proprio for concat in encode()
                # proprio should always be available (from obs or goal_proprio)
                late_p = proprio.float() if proprio is not None else None
                return _wrap(hidden, n_vid=N_vid, n_state=0, late_proprio=late_p)
            return _wrap(hidden)

        # --- State tokens ---
        state_tokens = self._embed_state_tokens(proprio.float())  # (B, N_state, D)
        N_state = state_tokens.shape[1]

        if N_state <= self.state_pos_emb.shape[1]:
            state_tokens = state_tokens + self.state_pos_emb[:, :N_state]
        state_tokens = state_tokens + self.modality_emb_state

        # --- Concat and forward ---
        all_tokens = torch.cat([vid_with_cls, state_tokens], dim=1)
        # (B, 1 + N_vid + N_state, D)

        hidden = self.vit.encoder(all_tokens).last_hidden_state
        hidden = self.vit.layernorm(hidden)
        return _wrap(hidden, n_vid=N_vid, n_state=N_state)


class _wrap:
    """Minimal output wrapper matching HF ViT's interface."""
    def __init__(self, last_hidden_state, n_vid=None, n_state=None, late_proprio=None):
        self.last_hidden_state = last_hidden_state
        self.n_vid = n_vid
        self.n_state = n_state
        self.late_proprio = late_proprio  # raw proprio for late_concat

try:
    from .grounding import PSGGroundingHeads, grounding_loss
except ImportError:
    from grounding import PSGGroundingHeads, grounding_loss


class CrossModalJEPA_v2(nn.Module):
    """JEPA with shared ViT that jointly encodes video + state patches.

    Encoder: CrossModalViTEncoder — one ViT forward for both modalities.
    Predictor: LeWM's ARPredictor on CLS tokens.
    Optional: grounding — PSGGroundingHeads (state + transition [+ optional velocity]);
              training-only, discarded at inference (see psgjepa/grounding.py).
    """

    def __init__(
        self,
        encoder: CrossModalViTEncoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        use_state: bool = True,
        grounding=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.use_state = use_state
        self.grounding = grounding

    def encode(self, info):
        pixels = info["pixels"].float()
        # Handle both (B, T, C, H, W) and (B, C, H, W) inputs
        added_time = False
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
            added_time = True
        B, T = pixels.shape[:2]
        pix_flat = rearrange(pixels, "b t c h w -> (b t) c h w")

        proprio_flat = None
        if self.use_state and "proprio" in info:
            proprio = info["proprio"].float()
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(1).expand(-1, T, -1)  # broadcast to match T
            proprio_flat = rearrange(proprio, "b t d -> (b t) d")

        out = self.encoder(pix_flat, proprio=proprio_flat)
        cls = out.last_hidden_state[:, 0]                   # CLS token
        # For late_concat: concat raw proprio to CLS before projector
        if out.late_proprio is not None:
            cls = torch.cat([cls, out.late_proprio], dim=-1)  # (B*T, D+proprio_dim)
        elif hasattr(self.encoder, 'state_inject_mode') and self.encoder.state_inject_mode == "late_concat":
            # Zero-pad when proprio missing (shouldn't happen if get_cost expands correctly)
            zero_p = torch.zeros(cls.shape[0], self.encoder.proprio_dim, device=cls.device)
            cls = torch.cat([cls, zero_p], dim=-1)
        emb = self.projector(cls)
        emb = rearrange(emb, "(b t) d -> b t d", b=B)
        # Always keep time dim — downstream (criterion, rollout) expects (B, T, D)
        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    # ----- Planning / inference methods (from original JEPA) -----

    def _strip_candidate_dim(self, info, num_samples):
        """Strip CEM candidate axis (S) only from tensors that have it."""
        out = {}
        for k, v in info.items():
            if not torch.is_tensor(v):
                continue
            if v.ndim >= 2 and v.shape[1] == num_samples:
                out[k] = v[:, 0]
            else:
                out[k] = v
        return out

    def rollout(self, info, action_sequence, history_size: int = 3):
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = self._strip_candidate_dim(info, S)
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            act = torch.cat([act, act_future[:, t:t+1]], dim=1)

        act_emb = self.action_encoder(act)
        pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        S = action_candidates.shape[1]
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # Ensure proprio/goal_proprio are expanded to match CEM's S dimension
        # CEM expands pixels to (B,S,T,C,H,W) but may not expand proprio
        if self.use_state:
            for key in ["proprio", "goal_proprio"]:
                if key in info_dict and torch.is_tensor(info_dict[key]):
                    p = info_dict[key]
                    if p.ndim >= 2 and (p.ndim < 3 or p.shape[1] != S):
                        # Add S dim: (B,T,D) → (B,S,T,D) or (B,D) → (B,S,D)
                        info_dict[key] = p.unsqueeze(1).expand(
                            *([p.shape[0], S] + list(p.shape[1:]))).contiguous()

        # Build goal dict: strip candidate dim, then swap pixels→goal, goal_X→X
        goal = self._strip_candidate_dim(info_dict, S)
        goal["pixels"] = goal.pop("goal", goal.get("pixels"))
        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_"):]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


# =====================================================================
# Training step (identical to LeWM's lejepa_forward)
# =====================================================================



# =====================================================================
# Training step: JEPA forward-prediction + SIGReg + physical state grounding
# =====================================================================

def psg_forward(self, batch, stage, cfg):
    """One training step. Bound onto the stable_pretraining LightningModule.

    Loss = L_JEPA (forward prediction) + lambda_reg * SIGReg + lambda_g * L_grounding,
    where L_grounding = L_static + L_dynamic (see psgjepa/grounding.py). The grounding heads
    live on ``self.model.grounding`` and are discarded at inference.
    """
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    if "proprio" in batch:
        batch["proprio"] = torch.nan_to_num(batch["proprio"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    # --- JEPA forward prediction + SIGReg anti-collapse (unchanged from LeWM) ---
    L = ctx_len + n_preds
    emb_j = emb[:, :L]
    ctx_emb = emb_j[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb_j[:, n_preds:]

    pred_emb = self.model.predict(ctx_emb, ctx_act)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # --- physical state grounding (training-only; heads discarded at inference) ---
    gcfg = cfg.loss.get("grounding", {}) if hasattr(cfg.loss, "get") else {}
    gw = float(gcfg.get("weight", 0.0))
    if self.model.grounding is not None and gw > 0.0:
        if "observation" not in batch:
            raise KeyError(
                "physical state grounding is enabled (loss.grounding.weight > 0) but the batch "
                "has no 'observation' column, so the grounding targets are unavailable. Add "
                "'observation' to data.dataset.keys_to_load, or set loss.grounding.weight=0.0 "
                "to train the plain LeWM baseline."
            )
        vel_idx = gcfg.get("vel_idx", None)
        gl = grounding_loss(
            self.model.grounding, emb, batch["observation"],
            state_idx=list(gcfg["state_idx"]),
            joint_idx=list(gcfg["joint_idx"]),
            vel_idx=list(vel_idx) if vel_idx is not None else None,
        )
        output["loss"] = output["loss"] + gw * gl["loss"]
        output["grounding_loss"] = gl["loss"].detach()
        for k in ("static", "djoint", "velocity"):
            if k in gl:
                output[f"grounding_{k}_loss"] = gl[k]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
