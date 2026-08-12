"""Physical state grounding for PSG-JEPA (training-only; heads are discarded at inference).

Two grounding objectives are added on top of the JEPA forward-prediction loss:

    L_static  : per-latent state grounding          H_s : z_t          -> s_t
    L_dynamic : latent-pair transition grounding
                  - multi-horizon joint-angle change H_d : (z_t, z_{t+k}) -> dq_{t,k}   (always)
                  - instantaneous joint velocity      H_v : (z_t, z_{t+1}) -> v_t        (optional)

    L_PSG = L_JEPA + lambda_g * (L_static + L_dynamic)

The velocity term is an optional component of the transition grounding: it is enabled only when
the environment logs a velocity signal (``use_velocity=True``). When the environment does not
provide velocity, transition grounding reduces to the multi-horizon joint-angle change alone, and
the method degrades gracefully to two heads with no other changes.

All heads are lightweight MLPs that share the frozen-at-inference encoder. They are used only to
shape the representation during world-model training and are dropped for deployment, so grounding
adds zero inference cost.
"""

import torch
from torch import nn


def _masked_mse(pred, tgt):
    """MSE over the finite entries of ``tgt``.

    Grounding targets come from the logged observation column, which is padded with NaN at
    episode boundaries in some datasets. A NaN there would poison the whole training loss
    (the JEPA path never touches this column, so nothing else would catch it). When the
    target is fully finite -- the case for the OGBench datasets used in the paper -- this
    takes the fast path and is identical to ``((pred - tgt) ** 2).mean()``.
    """
    finite = torch.isfinite(tgt)
    if bool(finite.all()):
        return ((pred - tgt) ** 2).mean()
    sq = (pred - torch.nan_to_num(tgt, 0.0, 0.0, 0.0)) ** 2
    return (sq * finite).sum() / finite.sum().clamp(min=1)


def _mlp(d_in, d_out, hidden=256):
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, d_out),
    )


class PSGGroundingHeads(nn.Module):
    """Grounding heads shared across all latents; discarded after world-model training.

    Args:
        embed_dim:    dimension of a single-frame latent z_t.
        state_dim:    dimension of the proprioceptive state target s_t (static grounding).
        joint_dim:    dimension of the joint-angle vector q_t (transition Delta-q target).
        vel_dim:      dimension of the velocity target (used only when ``use_velocity``).
        use_velocity: include the optional velocity term in transition grounding.
        hidden:       MLP hidden width.
    """

    def __init__(self, embed_dim, state_dim, joint_dim, vel_dim=None,
                 use_velocity=False, hidden=256):
        super().__init__()
        self.use_velocity = use_velocity
        self.state_head = _mlp(embed_dim, state_dim, hidden)           # H_s : z_t -> s_t
        self.djoint_head = _mlp(2 * embed_dim, joint_dim, hidden)      # H_d : (z_t, z_{t+k}) -> dq
        if use_velocity:
            assert vel_dim is not None, "use_velocity=True requires vel_dim"
            self.vel_head = _mlp(2 * embed_dim, vel_dim, hidden)       # H_v : (z_t, z_{t+1}) -> v_t
        else:
            self.vel_head = None


def grounding_loss(heads: PSGGroundingHeads, emb, state,
                   state_idx, joint_idx, vel_idx=None):
    """Compute L_static + L_dynamic for a batch.

    Args:
        heads:     a ``PSGGroundingHeads`` instance.
        emb:       (B, T, D) per-frame latents from the shared encoder.
        state:     (B, T, S) ground-truth proprioceptive observations (normalized), logged
                   alongside the video during world-model training.
        state_idx: column indices selecting the static state target s_t from ``state``.
        joint_idx: column indices selecting the joint-angle vector q_t (for the Delta-q target).
        vel_idx:   column indices selecting the velocity target (required iff use_velocity).

    Returns:
        dict with ``loss`` (scalar = L_static + L_dynamic) and detached per-term logs
        (``static``, ``djoint``, and ``velocity`` when enabled).
    """
    B, T, D = emb.shape

    # --- static grounding: each latent z_t -> proprioceptive state s_t ---
    s_tgt = state[..., state_idx]
    s_pred = heads.state_head(emb.reshape(B * T, D)).reshape(B, T, -1)
    l_static = _masked_mse(s_pred, s_tgt)

    # --- dynamic grounding (1): multi-horizon joint-angle change over all pairs (z_t, z_{t+k}) ---
    q = state[..., joint_idx]
    dj_terms = []
    for k in range(1, T):
        za = emb[:, :T - k].reshape(B * (T - k), D)
        zb = emb[:, k:].reshape(B * (T - k), D)
        dj_pred = heads.djoint_head(torch.cat([za, zb], dim=-1)).reshape(B, T - k, -1)
        dj_tgt = q[:, k:] - q[:, :T - k]
        dj_terms.append(_masked_mse(dj_pred, dj_tgt))
    l_djoint = torch.stack(dj_terms).mean()

    l_dynamic = l_djoint
    logs = {"static": l_static.detach(), "djoint": l_djoint.detach()}

    # --- dynamic grounding (2, optional): adjacent pair (z_t, z_{t+1}) -> instantaneous velocity ---
    if heads.use_velocity:
        assert vel_idx is not None, "use_velocity=True requires vel_idx"
        v_tgt = state[:, :-1][..., vel_idx]
        za = emb[:, :-1].reshape(B * (T - 1), D)
        zb = emb[:, 1:].reshape(B * (T - 1), D)
        v_pred = heads.vel_head(torch.cat([za, zb], dim=-1)).reshape(B, T - 1, -1)
        l_vel = _masked_mse(v_pred, v_tgt)
        l_dynamic = l_dynamic + l_vel
        logs["velocity"] = l_vel.detach()

    logs["loss"] = l_static + l_dynamic
    return logs
