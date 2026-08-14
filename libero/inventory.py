#!/usr/bin/env python3
"""The four LIBERO-Goal variants and the pretraining recipe behind each one.

Every variant is an encoder plus the OFT action head that was jointly fine-tuned on top of it,
evaluated with 50 rollouts on each of the 10 LIBERO-Goal tasks. `VARIANTS` is the single source
of truth for what distinguishes them: the hydra `data=` config and the loss overrides passed to
`train.py --config-name=psgjepa_libero`. `scripts/run_libero.sh` mirrors this table -- keep the two in sync.

Eval logs for all four variants are in `results/`; the published checkpoints are PSG-JEPA's.
Everything is training seed 3072.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RELEASED_SEED = 3072
RELEASED_VARIANTS = ("psgjepa",)   # checkpoints we publish; results/ covers all four
EVAL_SEED = 4242
PRETRAIN_EPOCH = 40
HEAD_EPOCH = 30


@dataclass(frozen=True)
class Variant:
    """One encoder pretraining recipe."""

    paper_name: str                # how the row is labelled in the paper
    data: str                      # hydra `data=` config ("" when there is no pretraining stage)
    grounding: str                 # what the grounding objective supervises
    overrides: list[str] = field(default_factory=list)   # extra `train.py` args
    encoder: str = "lewm"          # "lewm" or "dinov2_hf"


VARIANTS: dict[str, Variant] = {
    "lewm": Variant(
        paper_name="LeWM",
        data="libero_goal_cm", grounding="none (forward prediction + SIGReg only)",
        overrides=["loss.grounding.weight=0.0"],
    ),
    "lewm_actionidm": Variant(
        paper_name="LeWM_ActionIDM",
        data="libero_goal_cm", grounding="adjacent-pair action IDM only",
        overrides=["loss.grounding.weight=0.1", "loss.grounding.w_state=0.0",
                   "loss.grounding.w_djoint=0.0"],
    ),
    "dinov2": Variant(
        paper_name="DINOv2",
        data="", grounding="n/a -- pretrained facebook/dinov2-base used as the encoder",
        encoder="dinov2_hf",
    ),
    "psgjepa": Variant(
        paper_name="PSG-JEPA (ours)",
        data="libero_goal_cm_state",
        grounding="state + adjacent-pair action + multi-horizon delta-joint",
        overrides=["loss.grounding.weight=0.1"],
    ),
}


def release_paths(variant: str, seed: int = RELEASED_SEED) -> dict[str, str]:
    """Where a variant's artifacts live inside the released checkpoint bundle."""
    stem = f"{variant}/seed{seed}"
    paths = {"head": f"{stem}/oft_head_epoch{HEAD_EPOCH}.ckpt"}
    if VARIANTS[variant].encoder == "lewm":
        paths["encoder"] = f"{stem}/encoder_epoch{PRETRAIN_EPOCH}.ckpt"
    return paths


if __name__ == "__main__":
    for name, v in VARIANTS.items():
        print(f"{name:16s} {v.paper_name:16s} data={v.data or '-':22s} {v.grounding}")
        if v.overrides:
            print(f"{'':16s} overrides: {' '.join(v.overrides)}")
