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


# =====================================================================
# LIBERO-Goal variant: the motion target is the action, not the velocity
# =====================================================================
#
# LIBERO logs no joint velocity, so the transition head cannot be supervised on instantaneous
# velocity. The recorded action takes its place as the motion target -- the closest available
# signal for "what changed between these two latents" -- and the multi-horizon joint-angle term
# is unchanged. Selected with `loss.grounding.target: action`; see configs/psgjepa_libero.yaml.
#
#     L_static : H_s : z_t                -> s_t             robot-body state
#     L_motion : H_a : (z_t, z_{t+1})     -> a_t             the logged action
#     L_djoint : H_d : (z_t, z_{t+k})     -> q_{t+k} - q_t   every horizon k = 1 .. T-1
#
# Terms whose weight is zero are skipped, so a variant that uses only the motion head needs no
# state column in the dataset at all.

LIBERO_STATE_DIM = 15      # [joint_states(7), ee_pos(3), ee_ori(3), gripper_states(2)]
LIBERO_JOINT_DIM = 7       # leading joint_states block, the multi-horizon Delta-q target
LIBERO_ACTION_DIM = 7


class LiberoGroundingHeads(nn.Module):
    """Grounding heads for the LIBERO form; discarded after world-model training.

    Args:
        embed_dim:  dimension of a single-frame latent z_t.
        state_dim:  dimension of the robot-body state target s_t.
        joint_dim:  dimension of the joint vector q_t (the multi-horizon Delta-q target).
        action_dim: dimension of the logged action, the motion target.
        hidden:     MLP hidden width.
    """

    def __init__(self, embed_dim, state_dim=LIBERO_STATE_DIM, joint_dim=LIBERO_JOINT_DIM,
                 action_dim=LIBERO_ACTION_DIM, hidden=256):
        super().__init__()
        self.state_head = _mlp(embed_dim, state_dim, hidden)            # H_s : z_t -> s_t
        self.motion_head = _mlp(2 * embed_dim, action_dim, hidden)      # H_a : (z_t, z_{t+1}) -> a_t
        self.djoint_head = _mlp(2 * embed_dim, joint_dim, hidden)       # H_d : (z_t, z_{t+k}) -> dq


def libero_grounding_loss(heads, emb, action, state=None, joint_dim=LIBERO_JOINT_DIM,
                          w_state=0.1, w_motion=1.0, w_djoint=1.0):
    """Compute the LIBERO grounding loss for a batch.

    Args:
        heads:     a ``LiberoGroundingHeads`` instance.
        emb:       (B, T, D) per-frame latents from the shared encoder.
        action:    (B, T, A) logged actions; only the first T-1 steps are used.
        state:     (B, T, S) logged robot-body state. Required unless both ``w_state`` and
                   ``w_djoint`` are zero, in which case it is never touched.
        joint_dim: how many leading state dimensions form the joint vector q.

    Returns:
        dict with ``loss`` (the weighted sum) and detached per-term logs.
    """
    B, T, D = emb.shape
    logs = {}
    total = emb.new_zeros(())

    if w_state > 0 or w_djoint > 0:
        if state is None:
            raise ValueError("LIBERO grounding needs a state column when w_state or w_djoint > 0")

    # --- static grounding: every latent z_t must expose the robot-body state s_t ---
    if w_state > 0:
        state_pred = heads.state_head(emb.reshape(B * T, D)).reshape(B, T, -1)
        l_state = ((state_pred - state) ** 2).mean()
        total = total + w_state * l_state
        logs["static"] = l_state.detach()

    # --- motion grounding: the adjacent pair (z_t, z_{t+1}) must expose the action a_t ---
    if w_motion > 0:
        z_a = emb[:, :-1].reshape(B * (T - 1), D)
        z_b = emb[:, 1:].reshape(B * (T - 1), D)
        action_pred = heads.motion_head(torch.cat([z_a, z_b], dim=-1)).reshape(B, T - 1, -1)
        l_motion = ((action_pred - action[:, :T - 1]) ** 2).mean()
        total = total + w_motion * l_motion
        logs["motion"] = l_motion.detach()

    # --- transition grounding: every pair (z_t, z_{t+k}) must expose the net joint change ---
    if w_djoint > 0:
        q = state[..., :joint_dim]
        per_k = []
        for k in range(1, T):
            z_ak = emb[:, :T - k].reshape(B * (T - k), D)
            z_bk = emb[:, k:].reshape(B * (T - k), D)
            dq_pred = heads.djoint_head(torch.cat([z_ak, z_bk], dim=-1)).reshape(B, T - k, -1)
            per_k.append(((dq_pred - (q[:, k:] - q[:, :T - k])) ** 2).mean())
        l_djoint = sum(per_k) / len(per_k)
        total = total + w_djoint * l_djoint
        logs["djoint"] = l_djoint.detach()

    logs["loss"] = total
    return logs
