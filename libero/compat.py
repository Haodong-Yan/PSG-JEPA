#!/usr/bin/env python3
"""Make the released LIBERO encoder checkpoints loadable regardless of module layout.

The encoder checkpoints are whole-object `torch.save` dumps (`dump_object: True`), so the pickle
records the *module path* of every class it contains -- `cross_modal_v2.CrossModalJEPA_v2`,
`module.ARPredictor`, and so on. Those are the top-level module names the checkpoints were
trained under. This repository ships the same classes as `psgjepa.model` and `psgjepa.module`,
so unpickling fails with

    ModuleNotFoundError: No module named 'cross_modal_v2'

unless the old names are registered first. The class definitions are unchanged, only their home
moved, so aliasing the modules is enough -- no checkpoint rewriting is involved.

Two names need more than an alias. A whole-object dump also contains the training-only auxiliary
head that was attached to the model while it trained, under the name that head had at the time:
``TriDecompHead`` for the grounding variants and ``InverseDynamicsHead`` for the action-IDM one.
Those names have no counterpart in ``psgjepa.model``, so this module supplies them as thin
``nn.Module`` subclasses. Unpickling restores their weights into ``world_model.idm_head``; nothing
downstream reads it, and it is dropped when the encoder is handed to the action head.

Call `install()` before `torch.load`, or just import this module::

    import compat; compat.install()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)

`dataset.load_world_model()` already does this, so the train/eval entry points in this directory
need no extra step. This is only needed for the LeWM-family encoders; the DINOv2 rows load a
plain Hugging Face model and are unaffected.
"""

from __future__ import annotations

import importlib
import sys
import types

from torch import nn

# pickled module name -> module that provides the same classes here
ALIASES = {
    "cross_modal_v2": "psgjepa.model",
    "module": "psgjepa.module",
    "utils": "psgjepa.utils",
}


class TriDecompHead(nn.Module):
    """Training-only grounding head carried inside a whole-object encoder checkpoint.

    Present so the checkpoint unpickles; the weights land in ``world_model.idm_head`` and are
    unused at inference. The live implementation is ``psgjepa.grounding.LiberoGroundingHeads``.
    """


class InverseDynamicsHead(nn.Module):
    """Training-only action-IDM head carried inside a whole-object encoder checkpoint.

    Same story as ``TriDecompHead``: needed to unpickle, unused afterwards.
    """


LEGACY_HEADS = {"TriDecompHead": TriDecompHead, "InverseDynamicsHead": InverseDynamicsHead}


def _proxy(legacy_name: str, source) -> types.ModuleType:
    """A stand-in module exposing `source`'s classes plus the legacy training-only heads."""
    proxy = types.ModuleType(legacy_name)
    proxy.__dict__.update(source.__dict__)
    if legacy_name == "cross_modal_v2":
        proxy.__dict__.update(LEGACY_HEADS)
    return proxy


def install() -> list[str]:
    """Register aliases for any pickled module name that is not already importable.

    Returns the names that were aliased (empty when `cross_modal_v2` and `module` are already
    importable, in which case nothing needs redirecting).
    """
    aliased = []
    for legacy, modern in ALIASES.items():
        if legacy in sys.modules:
            continue
        try:
            importlib.import_module(legacy)
            continue                      # the original module is already on sys.path
        except ImportError:
            pass
        try:
            sys.modules[legacy] = _proxy(legacy, importlib.import_module(modern))
            aliased.append(legacy)
        except ImportError:
            # Neither layout is importable. Leave it alone so torch.load raises the real
            # ModuleNotFoundError, which names the missing module and is the useful error.
            pass
    return aliased


if __name__ == "__main__":
    print("aliased:", install() or "nothing (original modules are importable)")
