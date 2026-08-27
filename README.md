# PSG-JEPA

Official code for **"Is Forward Prediction Enough? Physical State Grounding for JEPA World
Models"** ([arXiv:2608.06799](https://arxiv.org/abs/2608.06799)).

PSG-JEPA keeps LeWM's JEPA world model (forward prediction + SIGReg anti-collapse) and adds a
lightweight **physical state grounding** objective. The grounding heads are used only during
training and are discarded at inference, so deployment keeps the original JEPA inference cost.

The repo covers the two evaluation tracks from the paper, both built on the same PSG-JEPA encoder:

- **Planning** on **OGBench-Cube** -- closed-loop success rate with a GC-IDM planner. Sections 1-3.
- **Policy** on **LIBERO-Goal** -- a task-conditioned OFT action head scored by simulator rollouts.
  Section 4, full details in [`libero/`](libero/).

## Layout
```
psgjepa/          # the method: cross-modal ViT + JEPA world model + grounding heads/loss
  model.py        #   encoder + JEPA world model + training step (psg_forward)
  grounding.py    #   PSGGroundingHeads + grounding_loss (OGBench and LIBERO forms)
  module.py       #   ARPredictor / SIGReg / MLP / Embedder (from LeWM)
  utils.py        #   dataset transforms + checkpoint callback
configs/          # hydra configs: psgjepa.yaml (OGBench), psgjepa_libero.yaml + data/
train.py          # encoder training entry point (both tracks)
scripts/train.sh  # OGBench-Cube encoder training
eval/             # GC-IDM planner (Nguyen et al., 2026) for the OGBench planning eval
libero/           # LIBERO-Goal policy learning: own README, released checkpoints, eval logs
```

## Install

Two Python 3.10 environments. The **planning** track (Sections 1-3) needs only Env A. The
**policy** track (Section 4) needs both, because the LIBERO simulator stack pins `numpy<2` while
the trainer needs `numpy>=2`, so they cannot share one environment.

**Env A -- world-model training + planning (OGBench):**
```bash
uv venv --python=3.10 .venv && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]==0.1.0" "stable-pretraining==0.1.6"
pip install -r requirements.txt
```

**Env B -- LIBERO policy (only for Section 4):** a separate `numpy<2` venv with the LIBERO
simulator. The exact steps are in [`libero/README.md`](libero/README.md#install).

Training is verified with `stable-worldmodel==0.1.0` (the paper numbers used `0.0.6`, equivalent).
Also tested with `torch==2.6.0`, `lightning==2.6.1`, `hydra-core==1.3.2`.

---

## 1. Data (OGBench-Cube)

We train on the OGBench single-expert **pixel** dataset `cube_single_expert` in the form released
with LeWM, not the state-based files from the [OGBench](https://github.com/seohongpark/ogbench)
repo. Download it from the LeWM HuggingFace collection
(<https://huggingface.co/collections/quentinll/lewm>) into the `stable_worldmodel` cache so that:
```
~/.stable_worldmodel/ogbench/cube_single_expert.h5
```
exists. Datasets resolve under `$STABLEWM_HOME` (default `~/.stable_worldmodel/`); set it to put
them elsewhere. Use the published `.h5` rather than re-rendering: OGBench renders differ slightly
across GPUs and our numbers were produced on the released files.

Arm observation layout (what the grounding target indices in `configs/psgjepa.yaml` refer to):
`joint_pos[0:6] | joint_vel[6:12] | effector+gripper[12:19] | privileged[19:]`.

## 2. Train the PSG-JEPA encoder

```bash
scripts/train.sh                 # seed 3072; = python train.py data=ogb_cm subdir=psgjepa_cube_seed3072
```
Config (`configs/psgjepa.yaml`): 10 epochs, batch 128, ViT-Tiny/14, `lambda_g=0.1`,
`sigreg.weight=0.09`, seed 3072. The epoch-10 checkpoint is written to:
```
~/.stable_worldmodel/psgjepa_cube_seed3072/psgjepa_epoch_10_object.ckpt
```
Ablation, no grounding (reduces to the LeWM baseline):
```bash
python train.py data=ogb_cm loss.grounding.weight=0.0
```

## 3. Planning eval (closed-loop success rate)

Closed-loop success is measured with the **GC-IDM planner** (Nguyen et al., 2026, code in
`eval/`, see Acknowledgments): extract latents from the frozen encoder, train a goal-conditioned
inverse-dynamics head, then run closed-loop planning.

```bash
export PYTHONPATH=$PWD:$PYTHONPATH        # so the checkpoint's psgjepa.model classes resolve
cd eval
CKPT=~/.stable_worldmodel/psgjepa_cube_seed3072/psgjepa_epoch_10_object.ckpt

# a) extract latents once
python train_idm.py extract --checkpoint $CKPT \
    --h5 ~/.stable_worldmodel/ogbench/cube_single_expert.h5 --output emb.npz

# b) one head per planner budget, then evaluate
for EP in 5 10 25 100; do
  python train_idm.py train --embeddings emb.npz --output gcidm_b${EP}.pt \
      --embed-dim 192 --action-dim 5 --epochs $EP --save-epochs $EP
  python eval_idm.py --dataset cube --checkpoint $CKPT --idm gcidm_b${EP}_ep${EP}.pt \
      --num-eval 200 --goal-offset 25 --eval-budget 50 --seed 42
done
```
We report success over n=200 goals, goal-offset 25, budget 50, averaged over 3 planner seeds
(42-44). A separate head is trained per budget because the LR schedule is
`CosineAnnealingLR(T_max=epochs)`, so a B-epoch head is fully annealed over B epochs rather than
being an epoch-B snapshot of a longer run. The limited-demonstration setting passes
`--train-split` to `train_idm.py train`.

---

## 4. Policy learning (LIBERO-Goal)

Pretrain the encoder on LIBERO-Goal demos with grounding, train a task-conditioned OFT action head
on top with the encoder jointly fine-tuned, then score by closed-loop rollouts in the LIBERO
simulator. Full instructions, released checkpoints and eval logs are in
[`libero/README.md`](libero/README.md). The short version:

**Reproduce from scratch** (needs Env B and the LIBERO-Goal demos):
```bash
# one-time: convert LIBERO demos to the HDF5 the encoder reads (writes the 15-d state target too)
python libero/convert_libero_to_swm.py --with-state --out-name libero_goal_cm_state

export PY_PRETRAIN=.venv/bin/python           # Env A (numpy>=2), encoder pretraining
export PY_LIBERO=.venv-libero/bin/python       # Env B (numpy<2), simulator + head
libero/scripts/run_libero.sh 3072              # pretrain -> train OFT head -> evaluate
```

**Or evaluate our released checkpoint** (no training):
```bash
python libero/weights/download_weights.py      # PSG-JEPA encoder + OFT head, 0.33 GiB
$PY_LIBERO libero/eval_oft_head.py \
    --checkpoint libero/weights/checkpoints/psgjepa/seed3072/oft_head_epoch30.ckpt \
    --tasks all --n-eval 50 --max-steps 600 --seed 4242 --output /tmp/psgjepa_seed3072.json
```
The released seed-3072 checkpoint scores **90.8%** mean success over the 10 tasks (50 rollouts
each). The paper's Table 4 reports the mean over three training seeds; this release ships one.

> OGBench-Scene planning: coming soon.

## Citation
```bibtex
@misc{yan2026forward,
  title         = {Is Forward Prediction Enough? Physical State Grounding for JEPA World Models},
  author        = {Haodong Yan and Jiaguan Zhu and Mingyuan Jia and Ruiqing Yin and Junjie He and Zhide Zhong and Junfeng Li and Jinxuan Lu and Hengtao Li and Tianran Zhang and Jiayi Chen and Wenxuan Song and Wen Chen and Yuxiang Gao and Haoang Li},
  year          = {2026},
  eprint        = {2608.06799},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

## Acknowledgments

PSG-JEPA is a grounding add-on built on top of **LeWorldModel (LeWM)**. Our code combines the
following codebases.

- **[LeWM](https://github.com/lucas-maes/le-wm)** (Maes et al., 2026) -- the JEPA world model this
  repo extends.
- **[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)** and
  **[stable-pretraining](https://github.com/galilai-group/stable-pretraining)** -- the training
  harness and OGBench environment/data wrappers.
- **[GC-IDM](https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models)**
  (Nguyen et al., 2026, [arXiv:2605.08732](https://arxiv.org/abs/2605.08732)) -- the amortized
  goal-conditioned inverse-dynamics planner in `eval/`, used for the closed-loop planning eval.
- **[OGBench](https://github.com/seohongpark/ogbench)** (Park et al., 2025) -- the cube benchmark
  and dataset.
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** (Liu et al., 2023) -- the
  LIBERO-Goal benchmark, simulator and demonstrations used in Section 4.
- **RC-aux** (Li et al., 2026, [arXiv:2605.07278](https://arxiv.org/abs/2605.07278)) -- the
  LIBERO-Goal OFT action head and evaluation protocol the policy track follows, for comparability.
