"""PCVRHyFormer training entry point (self-contained baseline).

Supports single-GPU and multi-GPU DDP training (via torchrun).

Usage:
    # Single GPU
    python train.py [--num_epochs 10] [--batch_size 256] ...
    # Multi-GPU
    torchrun --nproc_per_node=N train.py [--num_epochs 10] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


# ─────────────────────────── DDP Helpers ──────────────────────────────────


def setup_ddp():
    """Initialize DDP. Detects torchrun environment variables.

    Returns:
        (rank, local_rank, world_size). Non-DDP returns (0, 0, 1).
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                backend='nccl',
                device_id=torch.device(f'cuda:{local_rank}'),
            )
        except TypeError:
            dist.init_process_group(backend='nccl')
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ─────────────────────────────────────────────────────────────────────────


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")
    parser.add_argument('--experiment_name', type=str, default='PCVRHyFormer',
                        help='Human-readable experiment name for logs and summary blocks.')

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')
    parser.add_argument('--use_amp', dest='use_amp', action='store_true', default=True,
                        help='Enable CUDA automatic mixed precision training (default on)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                        help='Disable CUDA automatic mixed precision training')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help='AMP dtype, bf16 preferred on Ampere+ GPUs')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Enable torch.compile on model.forward')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--compile_skip_dynamic_cudagraphs',
                        dest='compile_skip_dynamic_cudagraphs',
                        action='store_true', default=True,
                        help='Skip CUDAGraph capture for dynamic input shapes '
                             'to avoid recording many graphs')
    parser.add_argument('--compile_allow_dynamic_cudagraphs',
                        dest='compile_skip_dynamic_cudagraphs',
                        action='store_false',
                        help='Allow CUDAGraph capture for dynamic input shapes')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='DataLoader prefetch_factor when num_workers > 0')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--full_train', action='store_true', default=False,
                        help='Use all Row Groups for training, disable validation/early stopping, '
                             'and save the final epoch checkpoint for submission.')
    parser.add_argument('--split_by_time', action='store_true',
                        help='Split train/valid by timestamp instead of Row Group position')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')
    parser.add_argument('--use_abs_time_user_int', action='store_true',
                        help='Append UTC+8 hour/weekday/hour-weekday synthetic user_int features')
    parser.add_argument('--use_sequence_summary_dense', action='store_true',
                        help='Append per-domain sequence activity and long-history summary dense features')
    parser.add_argument('--use_seq_trunc_dense', action='store_true',
                        help='Append per-domain sequence truncation summary dense features')
    parser.add_argument('--use_missing_indicator_dense', action='store_true',
                        help='Append missing/zero/empty summary dense features')
    parser.add_argument('--use_target_match_dense', action='store_true',
                        help='Append conservative target-history match dense features')
    parser.add_argument('--use_reserved_categorical_ids', action='store_true',
                        default=False,
                        help='Encode categorical ids as 0=padding, 1=missing/null, '
                             '2=oob_low/non-positive, 3=oob_high, 4+=raw_id+3, '
                             'and expand model vocab sizes accordingly.')
    parser.add_argument('--use_inter_event_gap_session', action='store_true',
                        help='V16_InterEventGapSession: add per-token inter-event gap and session-boundary embeddings')
    parser.add_argument('--use_mono_time_decay_attn_bias', action='store_true',
                        default=False,
                        help='V16_MonoTimeDecayAttnBias: add forced monotone '
                             'time-decay bias to sequence cross-attention scores.')
    parser.add_argument('--mono_time_decay_domains', type=str, default='seq_a,seq_b',
                        help='Comma-separated sequence domains receiving monotone '
                             'time-decay attention bias.')
    parser.add_argument('--mono_time_decay_lambda_min', type=float, default=0.02,
                        help='Minimum nonnegative decay lambda.')
    parser.add_argument('--mono_time_decay_lambda_init', type=float, default=0.02,
                        help='Initial total decay lambda before softplus learning.')
    parser.add_argument('--target_match_seq_fids', type=str, default='',
                        help="Target-match fids, e.g. 'seq_a:38;seq_b:70'. Empty keeps match features zero-filled.")
    parser.add_argument('--dense_robust_user_fids', type=str, default='',
                        help='V14_DenseRobust: comma-separated user_dense fids '
                             'to transform with nan-safe nonnegative log1p + '
                             'clip + optional scale. Empty disables it.')
    parser.add_argument('--dense_robust_clip', type=float, default=16.0,
                        help='Clip upper bound after log1p for '
                             '--dense_robust_user_fids.')
    parser.add_argument('--dense_robust_scale',
                        dest='dense_robust_scale',
                        action='store_true', default=True,
                        help='Scale clipped log1p dense values by '
                             '--dense_robust_clip (default on).')
    parser.add_argument('--no_dense_robust_scale',
                        dest='dense_robust_scale',
                        action='store_false',
                        help='Disable scaling after dense robust clipping.')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer', 'recent_core_tail'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--seq_encoder_type_by_domain', type=str, default='',
                        help="Optional per-domain encoder override, e.g. "
                             "'seq_a:transformer'. Unspecified domains use "
                             "--seq_encoder_type.")
    parser.add_argument('--seq_query_counts', type=str, default='',
                        help="V19_AsymSeqAHeavy: per-domain query allocation, "
                             "e.g. 'seq_a:4,seq_b:1,seq_c:1,seq_d:1'. "
                             "Unspecified domains use --num_queries.")
    parser.add_argument('--recent_core_k', type=int, default=64,
                        help='V17 recent_core_tail encoder: number of latest valid '
                             'tokens kept on the expensive self-attention path.')
    parser.add_argument('--recent_core_depth', type=int, default=3,
                        help='V17 recent_core_tail encoder: number of transformer '
                             'layers inside each short-window encoder.')
    parser.add_argument('--recent_core_tail', action='store_true', default=True,
                        help='V17 recent_core_tail encoder: prepend one mean-pooled '
                             'old-tail token before the recent core.')
    parser.add_argument('--no_recent_core_tail', action='store_false',
                        dest='recent_core_tail',
                        help='Disable the old-tail summary token.')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--per_token_ffn', action='store_true', default=False,
                        help='V14_PerTokenFFN: use RankMixer-style parameter-isolated '
                             'per-token FFN in Query Boosting (each token position owns '
                             'its own FFN weights). Default off = V14 shared-FFN baseline.')
    parser.add_argument('--ns_spatial_dropout_p', type=float, default=0.0,
                        help='Token-wise dropout probability for user/item NS tokens '
                             'during training only. Dense and sequence tokens are unchanged.')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--user_dense_tokens', type=int, default=1,
                        help='Number of projected user dense NS tokens')
    parser.add_argument('--item_dense_tokens', type=int, default=1,
                        help='Number of projected item dense NS tokens')
    # ── V14: SE-Net + training-trick bundle ──
    parser.add_argument('--use_se_net', action='store_true',
                        help='Enable SE-Net per-NS-token gating (V14).')
    parser.add_argument('--se_reduction', type=int, default=2,
                        help='Reduction ratio inside SE bottleneck MLP.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='AdamW weight_decay for dense params (V14: 0.02).')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        choices=['constant', 'cosine'],
                        help="Dense-LR schedule. 'cosine' uses warmup_steps + "
                             'cosine annealing down to 5%% of base LR.')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps for cosine schedule (V14: 500).')
    parser.add_argument('--use_ema', action='store_true',
                        help='Enable ModelEMA shadow of dense params; eval/save '
                             'use shadow weights.')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay factor (V14: 0.999).')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing magnitude (V14: 0.01).')
    parser.add_argument('--sample_weight_mode', type=str, default='none',
                        choices=['none', 'recency', 'montue', 'montue_hour'],
                        help='Per-row training loss weighting mode.')
    parser.add_argument('--sample_weight_recency_ref_ts', type=int, default=0,
                        help='Reference timestamp for recency weighting. '
                             '0 uses each batch max timestamp.')
    parser.add_argument('--sample_weight_recency_half_life_days', type=float, default=2.0,
                        help='Half-life in days for recency weighting.')
    parser.add_argument('--sample_weight_recency_max', type=float, default=1.5,
                        help='Raw max multiplier at ref_ts for recency weighting.')
    parser.add_argument('--sample_weight_montue_boost', type=float, default=1.4,
                        help='Raw multiplier for UTC+8 Monday/Tuesday rows.')
    parser.add_argument('--sample_weight_hour_strength', type=float, default=0.5,
                        help='Exponent applied to test-hour-distribution multipliers.')
    parser.add_argument('--sample_weight_min', type=float, default=0.5,
                        help='Minimum raw sample weight before batch mean normalization.')
    parser.add_argument('--sample_weight_max', type=float, default=2.0,
                        help='Maximum raw sample weight before batch mean normalization.')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    # ── DDP initialization ──
    rank, local_rank, world_size = setup_ddp()
    ddp_enabled = world_size > 1

    args = parse_args()

    # DDP: override device to local_rank.
    if ddp_enabled:
        args.device = f'cuda:{local_rank}'

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # Create output directories (only rank 0).
    if is_main_process():
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        if args.tf_events_dir:
            Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # DDP barrier: wait for rank 0 to finish creating directories.
    if ddp_enabled:
        dist.barrier()

    # Initialize logger and RNG.
    set_seed(args.seed + rank)  # Different seed per rank for data diversity.

    if is_main_process():
        create_logger(os.path.join(args.log_dir, 'train.log'))
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.full_train or (args.valid_ratio <= 0.0 and not args.split_by_time):
        args.full_train = True
        if args.split_by_time:
            logging.info("--full_train disables split_by_time")
        if args.train_ratio != 1.0:
            logging.info("--full_train forces train_ratio=1.0 (was %s)", args.train_ratio)
        if args.valid_ratio != 0.0:
            logging.info("--full_train forces valid_ratio=0.0 (was %s)", args.valid_ratio)
        if args.eval_every_n_steps != 0:
            logging.info(
                "--full_train disables eval_every_n_steps (was %s)",
                args.eval_every_n_steps,
            )
        args.split_by_time = False
        args.train_ratio = 1.0
        args.valid_ratio = 0.0
        args.eval_every_n_steps = 0

    logging.info(f"DDP: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    logging.info(f"Args: {vars(args)}")

    writer = None
    if is_main_process() and args.tf_events_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        full_train=args.full_train,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        buffer_batches=args.buffer_batches,
        seed=args.seed + rank,
        seq_max_lens=seq_max_lens,
        ddp_rank=rank,
        ddp_world_size=world_size,
        split_by_time=args.split_by_time,
        use_abs_time_user_int=args.use_abs_time_user_int,
        use_sequence_summary_dense=args.use_sequence_summary_dense,
        use_seq_trunc_dense=args.use_seq_trunc_dense,
        use_missing_indicator_dense=args.use_missing_indicator_dense,
        use_target_match_dense=args.use_target_match_dense,
        use_reserved_categorical_ids=args.use_reserved_categorical_ids,
        use_inter_event_gap_session=args.use_inter_event_gap_session,
        target_match_seq_fids=args.target_match_seq_fids,
        dense_robust_user_fids=args.dense_robust_user_fids,
        dense_robust_clip=args.dense_robust_clip,
        dense_robust_scale=args.dense_robust_scale,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "seq_encoder_type_by_domain": args.seq_encoder_type_by_domain,
        "seq_query_counts": args.seq_query_counts,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "user_dense_tokens": args.user_dense_tokens,
        "item_dense_tokens": args.item_dense_tokens,
        "ns_spatial_dropout_p": args.ns_spatial_dropout_p,
        # V16_InterEventGapSession
        "use_inter_event_gap_session": args.use_inter_event_gap_session,
        # V16_MonoTimeDecayAttnBias
        "use_mono_time_decay_attn_bias": args.use_mono_time_decay_attn_bias,
        "mono_time_decay_domains": args.mono_time_decay_domains,
        "mono_time_decay_lambda_min": args.mono_time_decay_lambda_min,
        "mono_time_decay_lambda_init": args.mono_time_decay_lambda_init,
        # V17_RecentCoreTail
        "recent_core_k": args.recent_core_k,
        "recent_core_depth": args.recent_core_depth,
        "recent_core_tail": args.recent_core_tail,
        # V14: SE-Net per-NS-token gating
        "use_se_net": args.use_se_net,
        "se_reduction": args.se_reduction,
        # V14_PerTokenFFN: parameter-isolated Query-Boosting FFN
        "per_token_ffn": args.per_token_ffn,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    if args.use_compile:
        if hasattr(torch, 'compile'):
            try:
                if args.compile_skip_dynamic_cudagraphs:
                    try:
                        import torch._inductor.config as inductor_config
                        inductor_config.triton.cudagraph_skip_dynamic_graphs = True
                        logging.info(
                            "torch.compile: skip dynamic-shape CUDAGraph capture")
                    except Exception:
                        logging.warning(
                            "Unable to set cudagraph_skip_dynamic_graphs; continuing")
                logging.info(
                    f"Compiling model.forward with torch.compile(mode={args.compile_mode})")
                model.forward = torch.compile(
                    model.forward, mode=args.compile_mode)
            except Exception:
                logging.exception("torch.compile failed; falling back to eager model")
        else:
            logging.warning("torch.compile is not available in this PyTorch build")

    # ── DDP wrapping ──
    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
        logging.info(f"Model wrapped with DDP on device cuda:{local_rank}")

    # Log model sizing info.
    raw_model = model.module if ddp_enabled else model
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = raw_model.num_ns
    total_q = getattr(raw_model, 'total_query_tokens', args.num_queries * num_sequences)
    T = total_q + num_ns
    logging.info(
        f"PCVRHyFormer model created: num_ns={num_ns}, "
        f"num_sequences={num_sequences}, num_queries={args.num_queries}, "
        f"total_query_tokens={total_q}, T={T}, d_model={args.d_model}, "
        f"rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        full_train=args.full_train,
        train_config=vars(args),
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        # V14: training-trick bundle
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        label_smoothing=args.label_smoothing,
        sample_weight_mode=args.sample_weight_mode,
        sample_weight_recency_ref_ts=args.sample_weight_recency_ref_ts,
        sample_weight_recency_half_life_days=args.sample_weight_recency_half_life_days,
        sample_weight_recency_max=args.sample_weight_recency_max,
        sample_weight_montue_boost=args.sample_weight_montue_boost,
        sample_weight_hour_strength=args.sample_weight_hour_strength,
        sample_weight_min=args.sample_weight_min,
        sample_weight_max=args.sample_weight_max,
    )

    trainer.train()

    if writer:
        writer.close()

    logging.info("Training complete!")
    cleanup_ddp()


if __name__ == "__main__":
    main()
