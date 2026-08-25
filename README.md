# TAAC 2026 UniRec Challenge — V19 single model

This repository releases the highest-scoring V-series single-model solution
from our final-round experiments for post-click CVR prediction. Given user
features, item features, and four behavior-sequence domains, the model
predicts whether a clicked impression converts.

> **Final-round public leaderboard AUC: 0.825625**
>
> Experiment: `V19_AsymSeqAHeavy_LongRecipe_NSDrop010` (evaluation ID
> `139174`, 2026-06-23).

The competition data, checkpoints, submission credentials, and platform
scripts are deliberately not included.

## Method overview

The model is a RankMixer/HyFormer-style architecture that creates tokens from
heterogeneous non-sequential features and retrieves information from four
behavior-sequence domains with cross-attention.

- **Structured tokenization.** `user_int`, `user_dense`, `item_int`, and
  `item_dense` are encoded as 5, 4, 2, and 1 tokens respectively. The tokens
  use independent projections; the Query Boosting block uses a parameter-
  isolated FFN for each token.
- **Time-aware representation.** The input pipeline adds UTC+8 hour,
  weekday, and hour×weekday features. It also buckets each historical event's
  age with non-uniform intervals and adds a time embedding to its sequence
  token.
- **Asymmetric sequence capacity.** The four sequence domains receive
  `seq_a:4, seq_b:1, seq_c:1, seq_d:1` query tokens. `seq_a` uses a
  Transformer encoder; the other domains use SwiGLU encoders. This allocates
  more capacity to the most useful sequence domain without increasing every
  sequence encoder equally.
- **Robust optimization.** Reserved categorical IDs distinguish padding,
  missing, and out-of-range values. The recipe also includes bf16 AMP,
  cosine warmup, EMA, label smoothing, SE-style token gating, DenseRobust,
  and non-sequential-token spatial dropout.
- **Training systems.** Parquet is streamed through a PyArrow
  `IterableDataset`; DDP uses `Join` to safely handle uneven data shards. The
  launcher adapts DataLoader and per-rank batch settings to CPU/GPU resources
  while preserving the intended global batch size across GPU counts.

## Repository layout

```
dataset.py   # PyArrow/Parquet streaming, feature construction, time buckets
model.py     # PCVRHyFormer model and tokenization modules
trainer.py   # training loop, AMP, EMA, DDP Join, checkpointing
train.py     # training entry point and configuration
infer.py     # checkpoint reconstruction and inference
run.sh       # V19 training recipe and multi-GPU launcher
```

## Setup

Python 3.10+ and CUDA-capable PyTorch are recommended. Install the PyTorch
wheel appropriate for your CUDA version first, then install the remaining
dependencies:

```bash
pip install -r requirements.txt
```

The code expects the competition-format Parquet directory and its
`schema.json`. No data is redistributed here; please obtain it under the
competition's terms.

## Train

Set explicit paths before launching. `run.sh` automatically uses `torchrun`
when multiple GPUs are visible.

```bash
export TRAIN_DATA_PATH=/path/to/train
export TRAIN_CKPT_PATH=/path/to/checkpoints
export TRAIN_LOG_PATH=/path/to/logs

# Default: final full-data recipe (four epochs).
bash run.sh
```

For a development split with validation and early stopping:

```bash
FULL_TRAIN=0 NUM_EPOCHS=8 PATIENCE=3 bash run.sh
```

Optional knobs include `CUDA_VISIBLE_DEVICES`, `GLOBAL_BATCH_SIZE`,
`NUM_WORKERS`, `PREFETCH_FACTOR`, and `USE_COMPILE=1`. `torch.compile` is
disabled by default because this asymmetric configuration previously showed
compile instability; enable it only after validating your own environment.

## Inference

`infer.py` rebuilds the model from the checkpoint's `train_config.json` and
uses the same feature construction as training.

```bash
export MODEL_OUTPUT_PATH=/path/to/checkpoint_dir
export EVAL_DATA_PATH=/path/to/test
export EVAL_RESULT_PATH=/path/to/predictions
python infer.py
```

## Reproducibility notes

The reported score is a public-leaderboard result from the stated competition
evaluation. Exact reproduction can vary with hardware, PyTorch/CUDA versions,
randomness, and the competition platform's data/runtime environment. This
repository contains the V19 code and recipe, but not model weights or data.

## License

Released under the [MIT License](LICENSE).
