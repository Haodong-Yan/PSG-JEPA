# PSG-JEPA on LIBERO-Goal

Policy-learning evaluation for **"Is Forward Prediction Enough? Physical State Grounding for JEPA
World Models"** ([arXiv:2608.06799](https://arxiv.org/abs/2608.06799)) — the LIBERO-Goal
policy-learning result of **Table 4**.

Each method pretrains an encoder on LIBERO-Goal demonstrations, then trains a task-conditioned OFT
action head on top of it while jointly fine-tuning the encoder, and is scored by closed-loop
rollouts in the LIBERO simulator. Every method shares the identical head, data pipeline and
evaluation protocol, so the differences come from the representation alone. The head is
conditioned on a 10-d task one-hot, not on language.

## Layout
```
inventory.py        # the four variants and the pretraining recipe behind each one
grounding.py        # the LIBERO grounding objective (state + action + multi-horizon Delta-q)
dataset.py          # LIBERO-Goal demo dataset, encoder loading, image transform
train_oft_head.py   # OFT action-head training with joint encoder fine-tuning
eval_oft_head.py    # closed-loop evaluation in the LIBERO simulator
convert_libero_to_swm.py   # LIBERO demos -> the HDF5 the encoder pretraining reads
compat.py           # module aliases so released checkpoints unpickle under any repo layout
scripts/run_libero.sh      # pretrain -> train head -> evaluate, for one variant and seed
results/            # one eval log per variant + make_table.py to summarize them
weights/            # manifest.json (sha256 of every released checkpoint) + download_weights.py
```

The two hydra dataset configs used by encoder pretraining live with the other data configs, at
`config/train/data/libero_goal_cm.yaml` and `config/train/data/libero_goal_cm_state.yaml`.

## Variants

| key | paper row | grounding objective | pretraining data |
|---|---|---|---|
| `lewm` | LeWM | none (forward prediction + SIGReg only) | `libero_goal_cm` |
| `lewm_actionidm` | LeWM<sub>ActionIDM</sub> | adjacent-pair action IDM only | `libero_goal_cm` |
| `dinov2` | DINOv2 | — (frozen-pretrained `facebook/dinov2-base` as encoder) | — |
| `psgjepa` | **PSG-JEPA (ours)** | state + adjacent-pair action + multi-horizon Δjoint | `libero_goal_cm_state` |

LIBERO logs no joint velocity, so the transition grounding uses the recorded **action** where
OGBench uses instantaneous velocity; the multi-horizon Δjoint term is unchanged. The grounding
target is the 15-d robot-body state `[joint_states(7), ee_pos(3), ee_ori(3), gripper(2)]`, with
the first 7 dimensions serving as the joint vector `q`. `grounding.py` holds that objective in
the same shape as the OGBench `psgjepa/grounding.py`; see `inventory.py` for the exact hydra
overrides behind each variant.

## Install

Two Python environments are required and cannot be merged: encoder pretraining runs against
`stable-worldmodel` (numpy >= 2), while the LIBERO simulator stack (robosuite / robomimic) pins
numpy < 2. Create both, and point `PY_PRETRAIN` / `PY_LIBERO` at them.

```bash
# 1. encoder pretraining environment (same as the OGBench half of this repo)
uv venv --python=3.10 .venv-pretrain && source .venv-pretrain/bin/activate
uv pip install "stable-worldmodel[train,env]==0.1.0" "stable-pretraining==0.1.6"

# 2. LIBERO environment
uv venv --python=3.10 .venv-libero && source .venv-libero/bin/activate
uv pip install "numpy<2" torch torchvision h5py easydict transformers
git clone https://github.com/Lifelong-Robot-Learning/LIBERO assets/benchmarks/LIBERO
uv pip install -e assets/benchmarks/LIBERO
```

Download the LIBERO-Goal demonstrations following the
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) instructions; the converter below
reads them from `assets/benchmarks/LIBERO`.

## Data

Encoder pretraining reads a single HDF5 file per dataset out of `$STABLEWM_HOME`:

```bash
# video + action only (used by lewm / lewm_actionidm)
python libero/convert_libero_to_swm.py --out-name libero_goal_cm
# additionally stores the 15-d robot state, the grounding target used by psgjepa
python libero/convert_libero_to_swm.py --with-state --out-name libero_goal_cm_state
```

Frames are preprocessed exactly as the action head sees them — rotate 180°, center-crop 0.9,
bilinear resize — so the pretrained encoder and the downstream head share one view geometry.

## Reproduce a row

```bash
export PY_PRETRAIN=.venv-pretrain/bin/python
export PY_LIBERO=.venv-libero/bin/python

libero/scripts/run_libero.sh psgjepa 3072      # pretrain -> train head -> evaluate
```

The script starts with a preflight that checks the dataset, the importable `libero` package and
the benchmark assets, so a missing dependency fails in seconds rather than after the hours of
pretraining that precede the first time the simulator is touched. It reads two LIBERO paths:
`LIBERO_ROOT`, the benchmark asset tree passed as `--libero-root`, and `LIBERO_PKG`, the
importable `libero` Python package. In a plain clone of the LIBERO repo these are the same
directory and the defaults are right; set `LIBERO_PKG` only if the assets and the package live
apart.

This runs encoder pretraining (40 epochs, ViT-Tiny/14, lr 5e-5, batch 64), then the OFT head
(30 epochs, hidden 1024, 4 layers, 8 heads, `seq_len=2` over `agentview_rgb` + `eye_in_hand_rgb`,
chunk 8, action horizon 8, lr 5e-5, batch 32, encoder jointly fine-tuned), then a closed-loop
evaluation of 50 rollouts on each of the 10 tasks with eval seed 4242 and `max_steps=600`.

Encoder pretraining and joint head training are both stochastic and GPU kernels are not
bit-deterministic, so a rerun tracks the released run closely rather than landing on it exactly.
Closed-loop evaluation carries its own spread of roughly ±1.5 points at a fixed eval seed.

## Results

`results/` holds one eval log per variant, from the released seed-3072 checkpoints. Summarize
them with:

```bash
python libero/results/make_table.py --per-task
```

| Method | Success (%) |
|---|---:|
| LeWM | 77.2 |
| LeWM<sub>ActionIDM</sub> | 85.2 |
| DINOv2 | 88.0 |
| **PSG-JEPA (ours)** | **90.8** |

Each number is the mean over the 10 task success rates, 50 rollouts per task. The paper's Table 4
reports the mean over three training seeds; this release ships one of them, so these are that
seed's numbers rather than the table's.

## Checkpoints

We release the **PSG-JEPA** checkpoints at training seed 3072: the pretrained encoder and the
OFT head that was jointly fine-tuned on it, 0.33 GiB together. `weights/manifest.json` lists both
with their sha256 and size. Evaluating the released head reproduces `results/psgjepa.json` to
within the eval spread noted above. The other three rows are reproduced by running the pipeline
above; their eval logs are in `results/` for comparison.

```bash
python libero/weights/download_weights.py                  # encoder + head, 0.33 GiB
python libero/weights/download_weights.py --kind encoder   # encoder only, 0.07 GiB
python libero/weights/download_weights.py --verify-only    # re-check local files
```

Evaluate a downloaded head directly:

```bash
$PY_LIBERO libero/eval_oft_head.py \
    --checkpoint libero/weights/checkpoints/psgjepa/seed3072/oft_head_epoch30.ckpt \
    --tasks all --n-eval 50 --max-steps 600 --seed 4242 --output /tmp/psgjepa_seed3072.json
```

The encoder checkpoints are whole-object pickles that record the module path of every class they
contain. `compat.py` registers the aliases those paths need, so a checkpoint loads under either
module layout — `dataset.load_world_model()` installs them before unpickling, so the entry points
here need no extra step.

## Acknowledgments

- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** (Liu et al., 2023) — the
  benchmark, simulator and demonstrations.
- **RC-aux** (Li et al., 2026, [arXiv:2605.07278](https://arxiv.org/abs/2605.07278)) — the
  LIBERO-Goal OFT action head and evaluation protocol this setup follows, so that our numbers are
  directly comparable to theirs.
- **[LeWM](https://github.com/lucas-maes/le-wm)** (Maes et al., 2026) and
  **[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)** — the world model
  PSG-JEPA extends and its training harness.
