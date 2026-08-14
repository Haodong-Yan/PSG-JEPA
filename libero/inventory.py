#!/usr/bin/env python3
"""The PSG-JEPA LIBERO-Goal recipe: what the encoder is pretrained with, and what is released.

The encoder is pretrained with physical state grounding, then an OFT action head is trained on
top of it with the encoder jointly fine-tuned, and the result is scored by 50 closed-loop
rollouts on each of the 10 LIBERO-Goal tasks. `scripts/run_libero.sh` runs those three stages
and mirrors the overrides below -- keep the two in sync.
"""

from __future__ import annotations

RELEASED_SEED = 3072
EVAL_SEED = 4242
PRETRAIN_EPOCH = 40
HEAD_EPOCH = 30

PAPER_NAME = "PSG-JEPA (ours)"
DATA = "libero_goal_cm_state"          # hydra `data=` config
GROUNDING = "state + adjacent-pair action + multi-horizon delta-joint"
OVERRIDES = ["loss.grounding.weight=0.1"]   # on top of configs/psgjepa_libero.yaml


def release_paths(seed: int = RELEASED_SEED) -> dict[str, str]:
    """Where the released artifacts live inside the checkpoint bundle."""
    stem = f"psgjepa/seed{seed}"
    return {
        "encoder": f"{stem}/encoder_epoch{PRETRAIN_EPOCH}.ckpt",
        "head": f"{stem}/oft_head_epoch{HEAD_EPOCH}.ckpt",
    }


if __name__ == "__main__":
    print(f"{PAPER_NAME}  seed {RELEASED_SEED}")
    print(f"  data       {DATA}")
    print(f"  grounding  {GROUNDING}")
    print(f"  overrides  {' '.join(OVERRIDES)}")
    for kind, path in release_paths().items():
        print(f"  {kind:9s}  {path}")
