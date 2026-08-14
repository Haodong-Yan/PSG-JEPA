# PSG-JEPA 

Official code for **"Is Forward Prediction Enough? Physical State Grounding for JEPA World
Models"** ([arXiv:2608.06799](https://arxiv.org/abs/2608.06799)).

PSG-JEPA keeps LeWM's JEPA world model (forward prediction + SIGReg anti-collapse) and adds a
lightweight **physical state grounding** objective. The grounding heads are used only during
training and are discarded at inference, so deployment keeps the original JEPA inference cost.

## Layout
```
psgjepa/
  model.py       # cross-modal ViT encoder + JEPA world model + training step (psg_forward)
  grounding.py   # PSGGroundingHeads + grounding_loss  (static + transition grounding)
  module.py      # ARPredictor / SIGReg / MLP / Embedder (from LeWM)
  utils.py       # dataset transforms + checkpoint callback
configs/         # hydra configs: psgjepa.yaml, psgjepa_libero.yaml + data/
train.py         # training entry point
scripts/         # train.sh
eval/            # GC-IDM planner (Nguyen et al., 2026) for the closed-loop planning eval
libero/          # LIBERO-Goal policy learning: OFT head, closed-loop eval, checkpoints
```

## Scope & roadmap

The release covers world-model training and goal-conditioned planning on OGBench-Cube --
training the PSG-JEPA encoder lives here, and closed-loop planning is run with the official GC-IDM
planner (see **Evaluate** below) -- and policy learning on LIBERO-Goal, which lives in
[`libero/`](libero/) with its own README, released checkpoints and evaluation logs.
OGBench-Scene: coming soon.

## Install
Set up the training environment (Python 3.10):
```bash
uv venv --python=3.10 && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]==0.1.0" "stable-pretraining==0.1.6"
pip install -r requirements.txt
```

## Data
We train on the OGBench single-expert **pixel** dataset `cube_single_expert` in the form released
with LeWM — not on the state-based files from the [OGBench](https://github.com/seohongpark/ogbench)
repo. Download it from the LeWM HuggingFace collection
(<https://huggingface.co/collections/quentinll/lewm>), which also hosts the pretrained LeWM
checkpoints.

`stable_worldmodel` resolves datasets under `$STABLEWM_HOME` (defaults to `~/.stable_worldmodel/`),
so the config name `ogbench/cube_single_expert` reads
`~/.stable_worldmodel/ogbench/cube_single_expert.h5`. Set `STABLEWM_HOME` to put them elsewhere.

Use the published `.h5` files rather than re-rendering the episodes: OGBench renders differ
slightly across GPUs, and our numbers were produced on the released files.

The arm observation layout, which is what the grounding target indices in
`configs/psgjepa.yaml` refer to:
`joint_pos[0:6] | joint_vel[6:12] | effector+gripper[12:19] | privileged[19:]`.

## Train
```bash
scripts/train.sh          # or: python train.py data=ogb_cm subdir=psgjepa_cube

# disable grounding entirely (= LeWM baseline):
python train.py data=ogb_cm loss.grounding.weight=0.0
```
Key config (`configs/psgjepa.yaml`): 10 epochs, batch 128, ViT-Tiny/14, `lambda_g=0.1`,
`sigreg.weight=0.09`, seed 3072. Grounding target indices live in the same file.

## Evaluate (goal-conditioned planning, closed-loop success rate)

Closed-loop success rate is measured with the **GC-IDM planner** of Nguyen et al. (2026), which
extracts latents from a frozen encoder, trains a goal-conditioned inverse-dynamics head, and runs
closed-loop planning. `eval/` is their code, see Acknowledgments.

```bash
# add THIS repo to PYTHONPATH so the checkpoint's `psgjepa.model` classes resolve when loaded:
export PYTHONPATH=/path/to/PSG-JEPA-release:$PYTHONPATH
cd eval
python train_idm.py extract --checkpoint <psgjepa_ckpt> --h5 ~/.stable_worldmodel/ogbench/cube_single_expert.h5 --output emb.npz

# one head per planner budget: the LR schedule is CosineAnnealingLR(T_max=epochs), so a
# B-epoch head is trained under a schedule fully annealed over B epochs -- NOT an epoch-B
# snapshot of a longer run. --save-epochs B writes the final-epoch weights (upstream's
# --output only keeps the best-val epoch).
for EP in 5 10 25 100; do
  python train_idm.py train --embeddings emb.npz --output gcidm_b${EP}.pt \
      --embed-dim 192 --action-dim 5 --epochs $EP --save-epochs $EP
  python eval_idm.py --dataset cube --checkpoint <psgjepa_ckpt> --idm gcidm_b${EP}_ep${EP}.pt \
      --num-eval 200 --goal-offset 25 --eval-budget 50 --seed 42
done
```
We report success rate over n=200 goals, goal-offset 25, budget 50, averaged over 3 planner seeds.
The limited-demonstration setting passes `--train-split` to `train_idm.py train`.

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

- **[LeWM](https://github.com/lucas-maes/le-wm)** (Maes et al., 2026) — the JEPA world model this
  repo extends.
- **[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)** and
  **[stable-pretraining](https://github.com/galilai-group/stable-pretraining)** — the training
  harness and OGBench environment/data wrappers.
- **[GC-IDM](https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models)**
  (Nguyen et al., 2026, [arXiv:2605.08732](https://arxiv.org/abs/2605.08732)) — the amortized
  goal-conditioned inverse-dynamics planner in `eval/`, used for the closed-loop planning
  evaluation.
- **[OGBench](https://github.com/seohongpark/ogbench)** (Park et al., 2025) — the cube benchmark
  and dataset.
