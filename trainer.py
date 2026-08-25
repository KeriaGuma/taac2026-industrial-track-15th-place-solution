"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Supports single-GPU and multi-GPU DDP training.

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import math
import os
import glob
import shutil
import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.algorithms.join import Join
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping, effective_rank, ModelEMA
from model import ModelInput


UTC8_OFFSET_SECONDS = 8 * 3600
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400

# Reference UTC+8 hour distribution used only when sample_weight_mode=montue_hour.
REFERENCE_HOUR_COUNTS = np.asarray([
    247090, 138239, 95486, 88900, 113207, 238855,
    432914, 493969, 550264, 552190, 586631, 610264,
    720817, 633371, 578244, 569752, 574881, 591827,
    638010, 786792, 891448, 899191, 762453, 456287,
], dtype=np.float64)


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def is_ddp() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


class _nullcontext:
    """Python 3.6 compatible null context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification (supports DDP).

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        full_train: bool = False,
        train_config: Optional[Dict[str, Any]] = None,
        use_amp: bool = True,
        amp_dtype: str = 'bf16',
        # ── V14: training-trick bundle ──
        weight_decay: float = 0.0,
        lr_schedule: str = 'constant',     # 'constant' | 'cosine'
        warmup_steps: int = 0,
        use_ema: bool = False,
        ema_decay: float = 0.0,
        label_smoothing: float = 0.0,
        sample_weight_mode: str = 'none',
        sample_weight_recency_ref_ts: int = 0,
        sample_weight_recency_half_life_days: float = 2.0,
        sample_weight_recency_max: float = 1.5,
        sample_weight_montue_boost: float = 1.4,
        sample_weight_hour_strength: float = 0.5,
        sample_weight_min: float = 0.5,
        sample_weight_max: float = 2.0,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: Optional[DataLoader] = valid_loader
        self.writer = writer
        self.schema_path: Optional[str] = schema_path
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Get the raw model (unwrap DDP if needed).
        self.raw_model = model.module if hasattr(model, 'module') else model

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        # Use raw_model's params (DDP wrapper points to the same storage).
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(self.raw_model, 'get_sparse_params'):
            sparse_params = self.raw_model.get_sparse_params()
            dense_params = self.raw_model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay
            )
            self._sparse_param_ptrs = {p.data_ptr() for p in sparse_params}
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                self.raw_model.parameters(), lr=lr, betas=(0.9, 0.98),
                weight_decay=weight_decay,
            )
            self._sparse_param_ptrs = set()

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.full_train = bool(full_train)
        if self.full_train and self.valid_loader is not None:
            logging.warning("full_train=True ignores the provided validation loader")
            self.valid_loader = None
        if self.valid_loader is None and not self.full_train:
            logging.warning(
                "No validation loader was provided; enabling full-train checkpoint mode."
            )
            self.full_train = True
        if self.full_train and self.eval_every_n_steps > 0:
            logging.info("full_train=True: disabling step-level validation")
            self.eval_every_n_steps = 0
        self.train_config: Optional[Dict[str, Any]] = train_config
        if self.train_config is not None:
            self.train_config['full_train'] = self.full_train
            if self.full_train:
                self.train_config['valid_ratio'] = 0.0
                self.train_config['eval_every_n_steps'] = 0
        self.use_amp = bool(use_amp) and str(device).startswith('cuda')
        self.amp_dtype_name = str(amp_dtype).lower()
        if self.amp_dtype_name not in ('bf16', 'fp16'):
            raise ValueError(f"Unsupported amp_dtype={amp_dtype!r}")
        if self.use_amp and self.amp_dtype_name == 'bf16':
            bf16_supported = getattr(torch.cuda, 'is_bf16_supported', lambda: True)()
            if not bf16_supported:
                logging.warning("bf16 AMP requested but not supported; falling back to fp16")
                self.amp_dtype_name = 'fp16'
        self.amp_dtype = (
            torch.bfloat16 if self.amp_dtype_name == 'bf16' else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=(self.use_amp and self.amp_dtype_name == 'fp16'))

        # ── V14: training-trick bundle wiring ──
        self.weight_decay = float(weight_decay)
        self.label_smoothing = float(label_smoothing)
        self.sample_weight_mode = str(sample_weight_mode).lower()
        self.sample_weight_recency_ref_ts = int(sample_weight_recency_ref_ts)
        self.sample_weight_recency_half_life_days = float(sample_weight_recency_half_life_days)
        self.sample_weight_recency_max = float(sample_weight_recency_max)
        self.sample_weight_montue_boost = float(sample_weight_montue_boost)
        self.sample_weight_hour_strength = float(sample_weight_hour_strength)
        self.sample_weight_min = float(sample_weight_min)
        self.sample_weight_max = float(sample_weight_max)
        if self.sample_weight_mode not in ('none', 'recency', 'montue', 'montue_hour'):
            raise ValueError(f"Unsupported sample_weight_mode={self.sample_weight_mode!r}")
        hour_counts = torch.tensor(REFERENCE_HOUR_COUNTS, dtype=torch.float32)
        self._test_hour_mult_cpu = hour_counts / hour_counts.sum().clamp_min(1.0) * 24.0
        self.warmup_steps = int(warmup_steps)
        self.lr_schedule = str(lr_schedule)
        self.dense_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        if self.lr_schedule == 'cosine':
            try:
                est_total_steps = self.num_epochs * train_loader.dataset.estimated_num_batches()
            except Exception:
                est_total_steps = max(self.num_epochs * 1000, 1)
            warmup_steps_ = self.warmup_steps

            def _lr_lambda(step: int) -> float:
                if step < warmup_steps_:
                    return step / max(1, warmup_steps_)
                progress = (step - warmup_steps_) / max(1, est_total_steps - warmup_steps_)
                return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

            self.dense_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.dense_optimizer, _lr_lambda)
            if is_main_process():
                logging.info(
                    f"V14 LR schedule: cosine, warmup_steps={warmup_steps_}, "
                    f"est_total_steps={est_total_steps}"
                )
        elif self.lr_schedule != 'constant':
            raise ValueError(f"Unsupported lr_schedule={self.lr_schedule!r}")

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema: Optional[ModelEMA] = None
        if self.use_ema and self.ema_decay > 0:
            self.ema = ModelEMA(
                self.raw_model,
                decay=self.ema_decay,
                exclude_param_ptrs=self._sparse_param_ptrs,
            )
            if is_main_process():
                logging.info(
                    f"V14 ModelEMA enabled: decay={self.ema_decay}, "
                    f"tracked_tensors={len(self.ema.shadow)}, "
                    f"excluded_sparse_tensors={len(self._sparse_param_ptrs)}"
                )

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"use_amp={self.use_amp}, amp_dtype={self.amp_dtype_name}, "
                     f"grad_scaler={self.scaler.is_enabled()}, "
                     f"weight_decay={self.weight_decay}, lr_schedule={self.lr_schedule}, "
                     f"warmup_steps={self.warmup_steps}, use_ema={self.use_ema}, "
                     f"ema_decay={self.ema_decay}, label_smoothing={self.label_smoothing}, "
                     f"sample_weight_mode={self.sample_weight_mode}, "
                     f"recency_ref_ts={self.sample_weight_recency_ref_ts}, "
                     f"recency_half_life_days={self.sample_weight_recency_half_life_days}, "
                     f"recency_max={self.sample_weight_recency_max}, "
                     f"montue_boost={self.sample_weight_montue_boost}, "
                     f"hour_strength={self.sample_weight_hour_strength}, "
                     f"weight_clip=[{self.sample_weight_min}, {self.sample_weight_max}]")

        # ── V9.1 target_match_dense monitoring ──
        # Locate the (offset, dim) of the 20-dim TARGET_MATCH block inside
        # user_dense_feats. 4 domains x 5 dims = 20. Per-domain layout:
        #   dim 0: hit (0/1)   dim 1: log1p(count)
        #   dim 2: head_lt20   dim 3: head_lt100
        #   dim 4: log1p(recency_gap_sec)
        self._tm_offset: Optional[int] = None
        self._tm_dim: int = 0
        self._tm_domains: list = []
        try:
            ds = train_loader.dataset
            offsets = getattr(ds, '_synthetic_user_dense_offsets', {}) or {}
            tm = offsets.get(2000005)  # TARGET_MATCH_DENSE_COL fid
            if tm is not None:
                self._tm_offset, self._tm_dim = int(tm[0]), int(tm[1])
                self._tm_domains = list(getattr(ds, 'seq_domains', []))[:4]
        except Exception as e:
            logging.warning(f"V9.1 target_match monitor init skipped: {e}")
        self._tm_train_history: list = []
        self._tm_valid_history: list = []
        self._tm_train_acc_cur: Optional[Dict[str, torch.Tensor]] = None
        self._tm_valid_acc_cur: Optional[Dict[str, torch.Tensor]] = None
        self._epoch_history: list = []  # list[dict] of {epoch, train_loss, val_auc, val_logloss}
        if is_main_process() and self._tm_offset is not None:
            logging.info(
                f"V9.1 target_match monitor: offset={self._tm_offset}, "
                f"dim={self._tm_dim}, domains={self._tm_domains}")
        elif is_main_process():
            logging.info("V9.1 target_match monitor: DISABLED "
                         "(use_target_match_dense=False or target_match_seq_fids empty)")

        # ── V14 sequence-usage monitor ──
        # Tracks per-domain sequence length distribution so seq-scaling
        # experiments are attributable. Three metrics per domain per epoch:
        #   * mean_len        — average valid_len entering the model
        #   * topk_sat_rate   — P(valid_len >= seq_top_k); high = top_k binds
        #   * trunc_rate      — P(valid_len >= max_len); high = max_len binds
        self._seq_domains_list: list = list(self._tm_domains)  # reuse domain order
        if not self._seq_domains_list:
            try:
                self._seq_domains_list = list(
                    getattr(train_loader.dataset, 'seq_domains', []))[:4]
            except Exception:
                self._seq_domains_list = []
        self._seq_top_k: int = int(
            (self.train_config or {}).get('seq_top_k', 0) or 0)
        # Per-domain max_len: from dataset (after seq_max_lens override).
        self._seq_max_len_per_domain: Dict[str, int] = {}
        try:
            ds = train_loader.dataset
            ml = getattr(ds, '_seq_maxlen', {}) or {}
            self._seq_max_len_per_domain = {k: int(v) for k, v in ml.items()}
        except Exception:
            pass
        self._seq_train_history: list = []
        self._seq_valid_history: list = []
        self._seq_train_acc_cur: Optional[Dict[str, torch.Tensor]] = None
        self._seq_valid_acc_cur: Optional[Dict[str, torch.Tensor]] = None
        if is_main_process():
            logging.info(
                f"V14 seq-usage monitor: domains={self._seq_domains_list}, "
                f"seq_top_k={self._seq_top_k}, "
                f"max_len_per_domain={self._seq_max_len_per_domain}")

        # ── V14_DenseRobust monitor ──
        self._dense_robust_blocks: list = []
        try:
            ds = train_loader.dataset
            self._dense_robust_blocks = list(
                getattr(ds, '_dense_robust_user_blocks', []) or [])
        except Exception:
            self._dense_robust_blocks = []
        self._dense_robust_train_history: list = []
        self._dense_robust_valid_history: list = []
        self._dense_robust_train_acc_cur: Optional[Dict[str, torch.Tensor]] = None
        self._dense_robust_valid_acc_cur: Optional[Dict[str, torch.Tensor]] = None
        if is_main_process():
            if self._dense_robust_blocks:
                logging.info(
                    f"V14_DenseRobust monitor: blocks={self._dense_robust_blocks}")
            else:
                logging.info("V14_DenseRobust monitor: DISABLED")

        # ── V14 throughput tracking ──
        # Per-epoch wall-clock samples/sec for train; useful when comparing
        # against baselines after changing seq scaling.
        self._epoch_throughput: list = []  # list[float] (samples/sec)

    def _tm_zero_acc(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._tm_offset is None or self._tm_dim <= 0:
            return None
        dev = torch.device(self.device)
        return {
            'sum': torch.zeros(self._tm_dim, dtype=torch.float64, device=dev),
            'nonzero_hit': torch.zeros(len(self._tm_domains), dtype=torch.float64, device=dev),
            'n': torch.zeros(1, dtype=torch.float64, device=dev),
        }

    def _tm_update(
        self, acc: Optional[Dict[str, torch.Tensor]], user_dense: torch.Tensor
    ) -> None:
        if acc is None or self._tm_offset is None:
            return
        slc = user_dense[:, self._tm_offset:self._tm_offset + self._tm_dim].detach().float()
        acc['sum'] += slc.sum(dim=0).double()
        for d_idx in range(len(self._tm_domains)):
            base = d_idx * 5
            if base < slc.shape[1]:
                acc['nonzero_hit'][d_idx] += (slc[:, base] > 0).sum().double()
        acc['n'] += slc.shape[0]

    def _tm_reduce(
        self, acc: Optional[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, Any]]:
        if acc is None:
            return None
        if is_ddp():
            dist.all_reduce(acc['sum'], op=dist.ReduceOp.SUM)
            dist.all_reduce(acc['nonzero_hit'], op=dist.ReduceOp.SUM)
            dist.all_reduce(acc['n'], op=dist.ReduceOp.SUM)
        n = float(acc['n'].item())
        if n <= 0:
            return None
        mean = (acc['sum'] / n).cpu().numpy().tolist()
        hits = acc['nonzero_hit'].cpu().numpy().tolist()
        rates = [h / n for h in hits]
        return {
            'n_samples': int(n),
            'dim_mean': mean,
            'domain_hit_count': hits,
            'domain_hit_rate': rates,
        }

    def _format_tm_block(
        self, label: str, stats: Optional[Dict[str, Any]]
    ) -> list:
        lines: list = []
        if stats is None:
            return lines
        lines.append(f"  [{label}] n_samples={stats['n_samples']:,}")
        for d_idx, domain in enumerate(self._tm_domains):
            rate = stats['domain_hit_rate'][d_idx]
            cnt = int(stats['domain_hit_count'][d_idx])
            base = d_idx * 5
            dm = stats['dim_mean']
            if base + 5 > len(dm):
                continue
            lines.append(
                f"    {domain:14s}  hit_rate={rate * 100:7.4f}%  hits={cnt:>10,}  "
                f"| mean: hit={dm[base + 0]:.6f}  log1p_cnt={dm[base + 1]:.6f}  "
                f"head20={dm[base + 2]:.6f}  head100={dm[base + 3]:.6f}  "
                f"log1p_recency={dm[base + 4]:.6f}"
            )
        return lines

    def _seq_zero_acc(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._seq_domains_list:
            return None
        dev = torch.device(self.device)
        n_dom = len(self._seq_domains_list)
        return {
            'sum_len': torch.zeros(n_dom, dtype=torch.float64, device=dev),
            'sum_len_sq': torch.zeros(n_dom, dtype=torch.float64, device=dev),
            'topk_sat': torch.zeros(n_dom, dtype=torch.float64, device=dev),
            'trunc': torch.zeros(n_dom, dtype=torch.float64, device=dev),
            'max_len': torch.zeros(n_dom, dtype=torch.float64, device=dev),
            'n': torch.zeros(1, dtype=torch.float64, device=dev),
        }

    def _seq_update(
        self, acc: Optional[Dict[str, torch.Tensor]], device_batch: Dict[str, Any]
    ) -> None:
        if acc is None or not self._seq_domains_list:
            return
        first_b: Optional[int] = None
        for d_idx, domain in enumerate(self._seq_domains_list):
            key = f'{domain}_len'
            if key not in device_batch:
                continue
            lens = device_batch[key].detach().float()
            if first_b is None:
                first_b = lens.shape[0]
            acc['sum_len'][d_idx] += lens.sum().double()
            acc['sum_len_sq'][d_idx] += (lens * lens).sum().double()
            ml = self._seq_max_len_per_domain.get(domain, 0)
            if self._seq_top_k > 0:
                acc['topk_sat'][d_idx] += (lens >= self._seq_top_k).sum().double()
            if ml > 0:
                acc['trunc'][d_idx] += (lens >= ml).sum().double()
            mx = lens.max().double()
            if mx > acc['max_len'][d_idx]:
                acc['max_len'][d_idx] = mx
        if first_b is not None:
            acc['n'] += first_b

    def _seq_reduce(
        self, acc: Optional[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, Any]]:
        if acc is None:
            return None
        if is_ddp():
            dist.all_reduce(acc['sum_len'], op=dist.ReduceOp.SUM)
            dist.all_reduce(acc['sum_len_sq'], op=dist.ReduceOp.SUM)
            dist.all_reduce(acc['topk_sat'], op=dist.ReduceOp.SUM)
            dist.all_reduce(acc['trunc'], op=dist.ReduceOp.SUM)
            dist.all_reduce(acc['max_len'], op=dist.ReduceOp.MAX)
            dist.all_reduce(acc['n'], op=dist.ReduceOp.SUM)
        n = float(acc['n'].item())
        if n <= 0:
            return None
        sum_len = acc['sum_len'].cpu().numpy()
        sum_len_sq = acc['sum_len_sq'].cpu().numpy()
        mean = (sum_len / n).tolist()
        # std via E[X^2] - E[X]^2 with clamp.
        var = (sum_len_sq / n) - (sum_len / n) ** 2
        var = var.clip(min=0.0)
        std = (var ** 0.5).tolist()
        sat = (acc['topk_sat'].cpu().numpy() / n).tolist()
        trunc = (acc['trunc'].cpu().numpy() / n).tolist()
        max_len_seen = acc['max_len'].cpu().numpy().tolist()
        return {
            'n_samples': int(n),
            'mean_len': mean,
            'std_len': std,
            'topk_sat_rate': sat,
            'trunc_rate': trunc,
            'max_len_seen': max_len_seen,
        }

    def _format_seq_block(
        self, label: str, stats: Optional[Dict[str, Any]]
    ) -> list:
        lines: list = []
        if stats is None or not self._seq_domains_list:
            return lines
        lines.append(f"  [{label}] n_samples={stats['n_samples']:,}  "
                     f"seq_top_k={self._seq_top_k}")
        for d_idx, domain in enumerate(self._seq_domains_list):
            ml = self._seq_max_len_per_domain.get(domain, 0)
            mn = stats['mean_len'][d_idx]
            sd = stats['std_len'][d_idx]
            sat = stats['topk_sat_rate'][d_idx] * 100
            tr = stats['trunc_rate'][d_idx] * 100
            mx = int(stats['max_len_seen'][d_idx])
            lines.append(
                f"    {domain:14s}  max_len={ml:>5}  mean={mn:>7.1f}  "
                f"std={sd:>7.1f}  max_seen={mx:>5}  "
                f"topk_sat={sat:>6.2f}%  trunc={tr:>6.2f}%"
            )
        return lines

    def _dense_robust_zero_acc(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._dense_robust_blocks:
            return None
        dev = torch.device(self.device)
        n = len(self._dense_robust_blocks)
        return {
            'sum': torch.zeros(n, dtype=torch.float64, device=dev),
            'sum_sq': torch.zeros(n, dtype=torch.float64, device=dev),
            'nonzero': torch.zeros(n, dtype=torch.float64, device=dev),
            'max': torch.zeros(n, dtype=torch.float64, device=dev),
            'n_values': torch.zeros(n, dtype=torch.float64, device=dev),
        }

    def _dense_robust_update(
        self, acc: Optional[Dict[str, torch.Tensor]], device_batch: Dict[str, Any]
    ) -> None:
        if acc is None or not self._dense_robust_blocks:
            return
        dense = device_batch.get('user_dense_feats')
        if dense is None:
            return
        dense = dense.detach().float()
        for i, (_fid, offset, length) in enumerate(self._dense_robust_blocks):
            if offset + length > dense.shape[1]:
                continue
            block = dense[:, offset:offset + length]
            acc['sum'][i] += block.sum().double()
            acc['sum_sq'][i] += (block * block).sum().double()
            acc['nonzero'][i] += (block > 0).sum().double()
            mx = block.max().double()
            if mx > acc['max'][i]:
                acc['max'][i] = mx
            acc['n_values'][i] += block.numel()

    def _dense_robust_reduce(
        self, acc: Optional[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, Any]]:
        if acc is None:
            return None
        if is_ddp():
            for key in ('sum', 'sum_sq', 'nonzero', 'n_values'):
                dist.all_reduce(acc[key], op=dist.ReduceOp.SUM)
            dist.all_reduce(acc['max'], op=dist.ReduceOp.MAX)
        n = acc['n_values'].cpu().numpy()
        if float(n.sum()) <= 0:
            return None
        total = acc['sum'].cpu().numpy()
        total_sq = acc['sum_sq'].cpu().numpy()
        mean = np.divide(total, np.maximum(n, 1.0))
        var = np.divide(total_sq, np.maximum(n, 1.0)) - mean ** 2
        var = var.clip(min=0.0)
        nonzero_rate = np.divide(
            acc['nonzero'].cpu().numpy(), np.maximum(n, 1.0))
        return {
            'mean': mean.tolist(),
            'std': (var ** 0.5).tolist(),
            'max': acc['max'].cpu().numpy().tolist(),
            'nonzero_rate': nonzero_rate.tolist(),
            'n_values': n.tolist(),
        }

    def _format_dense_robust_block(
        self, label: str, stats: Optional[Dict[str, Any]]
    ) -> list:
        lines: list = []
        if stats is None:
            return lines
        lines.append(f"  [{label}] transformed user_dense monitor")
        for i, (fid, _offset, length) in enumerate(self._dense_robust_blocks):
            lines.append(
                f"    fid={fid:<4d} dim={length:<4d} "
                f"mean={stats['mean'][i]:.6f} std={stats['std'][i]:.6f} "
                f"max={stats['max'][i]:.6f} "
                f"nonzero={stats['nonzero_rate'][i]*100:.3f}%"
            )
        return lines

    def _emit_final_summary(self) -> None:
        """V14: dump a clearly-bounded summary block so the tail of the job log
        contains everything needed to evaluate the experiment without grepping
        through epoch-by-epoch logs.

        Block markers match both V9.1 and V14 patterns so the run.sh tail-echo
        awk regex `(V9\\.1|V14)` picks it up either way."""
        sep = "=" * 80
        lines: list = []
        cfg = self.train_config or {}
        exp_name = str(cfg.get('experiment_name') or 'PCVRHyFormer')
        lines.append("")
        lines.append(sep)
        lines.append(f"=== {exp_name} FINAL TRAINING SUMMARY START ===")
        lines.append(sep)

        # Self-describing config snapshot (key knobs for attribution).
        lines.append(
            f"Experiment: seq_top_k={cfg.get('seq_top_k')}, "
            f"seq_max_lens={cfg.get('seq_max_lens') or '<default 256>'}, "
            f"seq_encoder_type={cfg.get('seq_encoder_type')}, "
            f"seq_encoder_type_by_domain={cfg.get('seq_encoder_type_by_domain') or '<none>'}, "
            f"seq_query_counts={cfg.get('seq_query_counts') or '<uniform>'}, "
            f"use_reserved_categorical_ids={cfg.get('use_reserved_categorical_ids')}, "
            f"full_train={cfg.get('full_train')}, "
            f"ns_spatial_dropout_p={cfg.get('ns_spatial_dropout_p')}, "
            f"use_inter_event_gap_session={cfg.get('use_inter_event_gap_session')}, "
            f"use_mono_time_decay_attn_bias={cfg.get('use_mono_time_decay_attn_bias')}, "
            f"recent_core_k={cfg.get('recent_core_k')}, "
            f"recent_core_depth={cfg.get('recent_core_depth')}, "
            f"recent_core_tail={cfg.get('recent_core_tail')}, "
            f"sample_weight_mode={cfg.get('sample_weight_mode')}, "
            f"d_model={cfg.get('d_model')}, "
            f"target_match_seq_fids={cfg.get('target_match_seq_fids') or '<none>'}, "
            f"num_epochs={cfg.get('num_epochs')}, patience={cfg.get('patience')}"
        )
        lines.append(f"Completed epochs: {len(self._epoch_history)}")

        # Per-epoch table.
        if self._epoch_history and self.full_train:
            lines.append("")
            lines.append("Per-epoch full-train results:")
            lines.append(
                f"  {'epoch':>5}  {'train_loss':>10}  "
                f"{'train_min':>9}  {'samp/s':>8}"
            )
            for r in self._epoch_history:
                lines.append(
                    f"  {r['epoch']:>5d}  {r['train_loss']:>10.6f}  "
                    f"{r['train_minutes']:>9.2f}  "
                    f"{r.get('train_throughput', 0.0):>8.1f}"
                )
            lines.append("")
            lines.append(
                "Full-train mode: validation/early stopping disabled; "
                "final EMA-shadow checkpoint is saved after the last epoch."
            )
        elif self._epoch_history:
            lines.append("")
            lines.append("Per-epoch results:")
            lines.append(
                f"  {'epoch':>5}  {'train_loss':>10}  {'val_auc':>9}  "
                f"{'val_logloss':>11}  {'train_min':>9}  {'eval_min':>8}  "
                f"{'samp/s':>8}"
            )
            for r in self._epoch_history:
                lines.append(
                    f"  {r['epoch']:>5d}  {r['train_loss']:>10.6f}  "
                    f"{r['val_auc']:>9.6f}  {r['val_logloss']:>11.6f}  "
                    f"{r['train_minutes']:>9.2f}  {r['eval_minutes']:>8.2f}  "
                    f"{r.get('train_throughput', 0.0):>8.1f}"
                )
            best = max(self._epoch_history, key=lambda r: r['val_auc'])
            lines.append("")
            lines.append(
                f"Best epoch: {best['epoch']}  "
                f"val_auc={best['val_auc']:.6f}  "
                f"val_logloss={best['val_logloss']:.6f}"
            )
            # Baseline overlay: if V14_BASELINE_AUC env var is a CSV like
            # "0.832,0.835,0.836,0.836,0.835" we print per-epoch delta. Useful
            # to compare V14 vs V9 mainline without leaving the log.
            baseline_env = os.environ.get('V14_BASELINE_AUC', '').strip()
            if baseline_env:
                try:
                    base_aucs = [float(x) for x in baseline_env.split(',') if x.strip()]
                    lines.append("")
                    lines.append(
                        f"Baseline overlay (V14_BASELINE_AUC=\"{baseline_env}\"):"
                    )
                    lines.append(
                        f"  {'epoch':>5}  {'val_auc':>9}  {'baseline':>9}  "
                        f"{'delta':>9}"
                    )
                    for i, r in enumerate(self._epoch_history):
                        if i < len(base_aucs):
                            delta = r['val_auc'] - base_aucs[i]
                            lines.append(
                                f"  {r['epoch']:>5d}  {r['val_auc']:>9.6f}  "
                                f"{base_aucs[i]:>9.6f}  {delta:>+9.6f}"
                            )
                except Exception as e:
                    lines.append(f"  (failed to parse V14_BASELINE_AUC: {e})")
        else:
            lines.append("(no completed epochs — training exited early)")

        # V14: sequence-usage summary.
        if self._seq_train_history:
            lines.append("")
            lines.append("Sequence usage (TRAIN, per domain):")
            lines.append(
                f"  Config: seq_top_k={self._seq_top_k}, "
                f"max_len={self._seq_max_len_per_domain}"
            )
            lines.append(
                f"  {'epoch':>5}  {'domain':14s}  {'mean':>7}  {'std':>7}  "
                f"{'max_seen':>8}  {'topk_sat':>9}  {'trunc':>7}"
            )
            for r in self._seq_train_history:
                ep = r['epoch']
                for d_idx, domain in enumerate(self._seq_domains_list):
                    lines.append(
                        f"  {ep:>5d}  {domain:14s}  "
                        f"{r['mean_len'][d_idx]:>7.1f}  "
                        f"{r['std_len'][d_idx]:>7.1f}  "
                        f"{int(r['max_len_seen'][d_idx]):>8d}  "
                        f"{r['topk_sat_rate'][d_idx]*100:>8.2f}%  "
                        f"{r['trunc_rate'][d_idx]*100:>6.2f}%"
                    )
            # Attribution hints based on saturation/truncation.
            last_seq = self._seq_train_history[-1]
            hints: list = []
            for d_idx, domain in enumerate(self._seq_domains_list):
                sat = last_seq['topk_sat_rate'][d_idx]
                tr = last_seq['trunc_rate'][d_idx]
                if sat < 0.10 and self._seq_top_k > 0:
                    hints.append(
                        f"  - {domain}: topk_sat={sat*100:.2f}% → seq_top_k "
                        f"is OVER-PROVISIONED (could shrink without info loss)"
                    )
                elif sat > 0.50 and tr < 0.10:
                    hints.append(
                        f"  - {domain}: topk_sat={sat*100:.2f}% trunc={tr*100:.2f}% "
                        f"→ raising seq_top_k may yield more"
                    )
                elif tr > 0.20:
                    hints.append(
                        f"  - {domain}: trunc={tr*100:.2f}% → max_len is the "
                        f"binding constraint (raise it before top_k)"
                    )
            if hints:
                lines.append("")
                lines.append("Sequence-scaling attribution hints:")
                lines.extend(hints)

        # Target-match per-epoch hit rates.
        if self._tm_offset is not None:
            lines.append("")
            lines.append(
                f"Target-match monitor: offset={self._tm_offset}, "
                f"dim={self._tm_dim}, domains={self._tm_domains}"
            )
            lines.append("Expected: only seq_c hit rate should be non-trivial "
                         "(historical baseline ~0.253%).")
            lines.append("")
            for r in self._tm_train_history:
                ep = r['epoch']
                rates = r.get('domain_hit_rate', [])
                rate_str = "  ".join(
                    f"{d}={rates[i]*100:.4f}%"
                    for i, d in enumerate(self._tm_domains) if i < len(rates)
                )
                lines.append(f"  TRAIN epoch {ep}: n={r['n_samples']:,}  {rate_str}")
            for r in self._tm_valid_history:
                ep = r['epoch']
                rates = r.get('domain_hit_rate', [])
                rate_str = "  ".join(
                    f"{d}={rates[i]*100:.4f}%"
                    for i, d in enumerate(self._tm_domains) if i < len(rates)
                )
                lines.append(f"  VALID epoch {ep}: n={r['n_samples']:,}  {rate_str}")

            # Per-dim means at the last epoch (5 dims per domain).
            if self._tm_train_history:
                last = self._tm_train_history[-1]
                lines.append("")
                lines.append(
                    f"Final epoch ({last['epoch']}) TRAIN per-dim means "
                    f"(hit / log1p_cnt / head20 / head100 / log1p_recency):"
                )
                dm = last['dim_mean']
                for d_idx, domain in enumerate(self._tm_domains):
                    base = d_idx * 5
                    if base + 5 > len(dm):
                        continue
                    lines.append(
                        f"  {domain:14s}  "
                        f"{dm[base + 0]:.6f}  {dm[base + 1]:.6f}  "
                        f"{dm[base + 2]:.6f}  {dm[base + 3]:.6f}  {dm[base + 4]:.6f}"
                    )

            # Sanity check: did seq_c trigger?
            if self._tm_train_history and 'domain_c_seq' in self._tm_domains:
                idx = self._tm_domains.index('domain_c_seq')
                rates = [r['domain_hit_rate'][idx] * 100 for r in self._tm_train_history
                         if idx < len(r.get('domain_hit_rate', []))]
                if rates:
                    lines.append("")
                    lines.append(
                        f"SANITY: domain_c_seq train hit_rate per epoch = "
                        f"{['%.4f%%' % r for r in rates]}"
                    )
                    if max(rates) < 0.01:
                        lines.append(
                            "  WARNING: hit_rate < 0.01% — target_match likely "
                            "NOT activated. Check --target_match_seq_fids wiring."
                        )
        else:
            lines.append("")
            lines.append("Target-match monitor was DISABLED for this run.")

        lines.append("")
        if self._dense_robust_blocks:
            lines.append("V14_DenseRobust transformed user_dense monitor:")
            lines.append(
                f"  config: fids={(self.train_config or {}).get('dense_robust_user_fids')}, "
                f"clip={(self.train_config or {}).get('dense_robust_clip')}, "
                f"scale={(self.train_config or {}).get('dense_robust_scale')}"
            )
            lines.append(
                f"  {'split':>5}  {'epoch':>5}  {'fid':>5}  {'mean':>10}  "
                f"{'std':>10}  {'max':>10}  {'nonzero':>9}"
            )
            for split, hist in (
                ('train', self._dense_robust_train_history),
                ('valid', self._dense_robust_valid_history),
            ):
                for r in hist:
                    for i, (fid, _offset, _length) in enumerate(self._dense_robust_blocks):
                        lines.append(
                            f"  {split:>5}  {r['epoch']:>5d}  {fid:>5d}  "
                            f"{r['mean'][i]:>10.6f}  {r['std'][i]:>10.6f}  "
                            f"{r['max'][i]:>10.6f}  "
                            f"{r['nonzero_rate'][i]*100:>8.3f}%"
                        )
        else:
            lines.append("V14_DenseRobust monitor: DISABLED")

        lines.append("")
        lines.append(sep)
        lines.append(f"=== {exp_name} FINAL TRAINING SUMMARY END ===")
        lines.append(sep)

        text = "\n".join(lines)
        # Print to stdout and emit via logging so it lands wherever logs go.
        print(text, flush=True)
        for ln in lines:
            logging.info(ln)

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.amp.autocast('cuda', dtype=self.amp_dtype)

    def _build_step_dir_name(
        self,
        global_step: int,
        is_best: bool = False,
        is_full_train: bool = False,
    ) -> str:
        """Build a checkpoint sub-directory name."""
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_full_train:
            name += ".full_train"
        elif is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``."""
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        is_full_train: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save model.pt plus sidecar files (only on rank 0)."""
        if not is_main_process():
            return ""
        dir_name = self._build_step_dir_name(
            global_step, is_best=is_best, is_full_train=is_full_train)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories."""
        if not is_main_process():
            return
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _remove_old_full_train_dirs(self) -> None:
        """Delete stale ``*.full_train`` directories."""
        if not is_main_process():
            return
        pattern = os.path.join(self.save_dir, "global_step*.full_train")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old full_train dir: {old_dir}")

    def _save_full_train_checkpoint(self, global_step: int) -> str:
        """Save the final full-data checkpoint in the infer-compatible layout."""
        self._remove_old_full_train_dirs()
        if self.ema is not None:
            self.ema.apply_shadow(self.raw_model)
        try:
            ckpt_dir = self._save_step_checkpoint(
                global_step, is_full_train=True)
            if ckpt_dir:
                self.early_stopping.checkpoint_path = os.path.join(
                    ckpt_dir, "model.pt")
                logging.info(
                    "Saved full-train final checkpoint to %s/model.pt",
                    ckpt_dir,
                )
            return ckpt_dir
        finally:
            if self.ema is not None:
                self.ema.restore(self.raw_model)

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in batch to self.device."""
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically (only on rank 0)."""
        if not is_main_process():
            return

        # V14 EMA FIX: evaluate() measures val_auc on the EMA *shadow* weights
        # but restores the raw training weights in its finally block. The
        # checkpoint persisted below snapshots self.raw_model, so without
        # re-applying the shadow here we would write raw (non-EMA) weights to
        # model.pt while the reported CV reflects the EMA model -> the deployed
        # checkpoint and the CV number describe two different models. Re-apply
        # the shadow for the duration of checkpoint persistence (disk
        # state_dict + in-memory best_model deepcopy), then restore the raw
        # training weights so the next optimizer step keeps updating the raw
        # params and the EMA accumulator stays well-defined. Sparse Embedding
        # params are excluded from EMA, so this swap leaves them untouched.
        # When EMA is disabled (self.ema is None) this is a no-op and behavior
        # is identical to before.
        if self.ema is not None:
            self.ema.apply_shadow(self.raw_model)
        try:
            old_best = self.early_stopping.best_score
            is_likely_new_best = (
                old_best is None
                or val_auc > old_best + self.early_stopping.delta
            )
            if not is_likely_new_best:
                self.early_stopping(val_auc, self.raw_model, {
                    "best_val_AUC": val_auc,
                    "best_val_logloss": val_logloss,
                })
                return

            best_dir = os.path.join(
                self.save_dir,
                self._build_step_dir_name(total_step, is_best=True),
            )
            self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
            self._remove_old_best_dirs()

            self.early_stopping(val_auc, self.raw_model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })

            if self.early_stopping.best_score != old_best and os.path.exists(
                self.early_stopping.checkpoint_path
            ):
                self._save_step_checkpoint(
                    total_step, is_best=True, skip_model_file=True)
        finally:
            if self.ema is not None:
                self.ema.restore(self.raw_model)

    def _broadcast_early_stop(self) -> bool:
        """Broadcast rank 0's early_stop flag to all ranks. Returns whether to stop."""
        if not is_ddp():
            return self.early_stopping.early_stop

        flag = torch.tensor(
            [1 if (is_main_process() and self.early_stopping.early_stop) else 0],
            dtype=torch.long, device=self.device
        )
        dist.broadcast(flag, src=0)
        return flag.item() == 1

    def train(self) -> None:
        """Main training loop with DDP support."""
        if is_main_process():
            print("Start training (PCVRHyFormer)")
        try:
            self._train_impl()
        finally:
            if is_main_process():
                self._emit_final_summary()

    def _train_impl(self) -> None:
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            num_batches = self.train_loader.dataset.estimated_num_batches()
            loss_sum = 0.0
            epoch_t0 = time.time()
            # V9.1: reset per-epoch target_match accumulator.
            self._tm_train_acc_cur = self._tm_zero_acc()
            # V14: reset per-epoch seq-usage accumulator.
            self._seq_train_acc_cur = self._seq_zero_acc()
            # V14_DenseRobust: reset transformed dense monitor accumulator.
            self._dense_robust_train_acc_cur = self._dense_robust_zero_acc()

            # Join: ensures ranks that finish their data early participate in
            # empty all-reduce calls, preventing DDP deadlock with IterableDataset.
            join_ctx = Join([self.model]) if is_ddp() else _nullcontext()
            with join_ctx:
                for step, batch in enumerate(self.train_loader):
                    loss = self._train_step(batch)
                    total_step += 1
                    loss_sum += loss

                    if self.writer and is_main_process():
                        self.writer.add_scalar('Loss/train', loss, total_step)

                    if is_main_process() and (step + 1) % 100 == 0:
                        avg_loss = loss_sum / (step + 1)
                        elapsed = time.time() - epoch_t0
                        speed = (step + 1) / elapsed
                        eta = (num_batches - step - 1) / speed if speed > 0 else 0
                        print(f"Epoch {epoch} | step {step+1}/{num_batches} | "
                              f"loss={loss:.4f} avg={avg_loss:.4f} | "
                              f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s | "
                              f"global_step={total_step}")

                    # Step-level validation (only when eval_every_n_steps > 0).
                    if (
                        not self.full_train
                        and self.eval_every_n_steps > 0
                        and total_step % self.eval_every_n_steps == 0
                    ):
                        logging.info(f"Evaluating at step {total_step}")
                        val_auc, val_logloss = self.evaluate(epoch=epoch)
                        self.model.train()
                        torch.cuda.empty_cache()

                        if is_main_process():
                            logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")
                            if self.writer:
                                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                        self._handle_validation_result(total_step, val_auc, val_logloss)

                        should_stop = self._broadcast_early_stop()
                        if should_stop:
                            logging.info(f"Early stopping at step {total_step}")
                            return

            train_elapsed = time.time() - epoch_t0
            if is_main_process():
                logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / max(num_batches, 1):.4f}, "
                             f"train time: {train_elapsed/60:.1f}min")

            if self.full_train:
                eval_elapsed = 0.0
                val_auc = float('nan')
                val_logloss = float('nan')
                valid_tm_stats = None
                valid_seq_stats = None
                valid_dense_robust_stats = None
                self._tm_valid_acc_cur = None
                self._seq_valid_acc_cur = None
                self._dense_robust_valid_acc_cur = None
            else:
                eval_t0 = time.time()
                val_auc, val_logloss = self.evaluate(epoch=epoch)
                eval_elapsed = time.time() - eval_t0
                self.model.train()
                torch.cuda.empty_cache()

            # V9.1: reduce per-epoch target_match stats (DDP-aware) and append
            # to history. Done on all ranks because _tm_reduce uses all_reduce.
            train_tm_stats = self._tm_reduce(self._tm_train_acc_cur)
            if not self.full_train:
                valid_tm_stats = self._tm_reduce(self._tm_valid_acc_cur)
            self._tm_train_acc_cur = None
            self._tm_valid_acc_cur = None
            # V14: reduce per-epoch seq-usage stats (DDP-aware).
            train_seq_stats = self._seq_reduce(self._seq_train_acc_cur)
            if not self.full_train:
                valid_seq_stats = self._seq_reduce(self._seq_valid_acc_cur)
            self._seq_train_acc_cur = None
            self._seq_valid_acc_cur = None
            train_dense_robust_stats = self._dense_robust_reduce(
                self._dense_robust_train_acc_cur)
            if not self.full_train:
                valid_dense_robust_stats = self._dense_robust_reduce(
                    self._dense_robust_valid_acc_cur)
            self._dense_robust_train_acc_cur = None
            self._dense_robust_valid_acc_cur = None
            # V14: per-epoch train throughput (samples/sec). DDP-aware via
            # multiplying main-rank rate by world_size since each rank
            # processes its own shard at similar rate.
            if train_seq_stats is not None and train_elapsed > 0:
                train_thru = train_seq_stats['n_samples'] / train_elapsed
            else:
                train_thru = 0.0

            if is_main_process():
                if self.full_train:
                    print(f"Epoch {epoch} done | train {train_elapsed/60:.1f}min | "
                          "full_train: eval skipped")
                    logging.info(
                        f"Epoch {epoch} FullTrain | validation skipped")
                else:
                    print(f"Epoch {epoch} done | train {train_elapsed/60:.1f}min | "
                          f"eval {eval_elapsed/60:.1f}min")
                    logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")
                if self.writer and not self.full_train:
                    self.writer.add_scalar('AUC/valid', val_auc, total_step)
                    self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                epoch_record = {
                    'epoch': epoch,
                    'train_loss': loss_sum / max(num_batches, 1),
                    'val_auc': val_auc,
                    'val_logloss': val_logloss,
                    'train_minutes': train_elapsed / 60.0,
                    'eval_minutes': eval_elapsed / 60.0,
                    'train_throughput': train_thru,  # V14
                }
                self._epoch_history.append(epoch_record)
                self._epoch_throughput.append(train_thru)
                if train_tm_stats is not None:
                    self._tm_train_history.append({'epoch': epoch, **train_tm_stats})
                if valid_tm_stats is not None:
                    self._tm_valid_history.append({'epoch': epoch, **valid_tm_stats})
                for line in self._format_tm_block(f"Epoch {epoch} TRAIN target_match", train_tm_stats):
                    logging.info(line)
                for line in self._format_tm_block(f"Epoch {epoch} VALID target_match", valid_tm_stats):
                    logging.info(line)
                if self.writer and train_tm_stats is not None:
                    for d_idx, domain in enumerate(self._tm_domains):
                        self.writer.add_scalar(
                            f'TargetMatch/train_hit_rate/{domain}',
                            train_tm_stats['domain_hit_rate'][d_idx], total_step)
                if self.writer and valid_tm_stats is not None:
                    for d_idx, domain in enumerate(self._tm_domains):
                        self.writer.add_scalar(
                            f'TargetMatch/valid_hit_rate/{domain}',
                            valid_tm_stats['domain_hit_rate'][d_idx], total_step)

                # V14: record + log per-epoch seq-usage block + throughput.
                logging.info(
                    f"Epoch {epoch} Throughput | "
                    f"train_samples_per_sec={train_thru:.1f} "
                    f"(train_minutes={train_elapsed/60:.2f})"
                )
                if train_seq_stats is not None:
                    self._seq_train_history.append({'epoch': epoch, **train_seq_stats})
                if valid_seq_stats is not None:
                    self._seq_valid_history.append({'epoch': epoch, **valid_seq_stats})
                for line in self._format_seq_block(f"Epoch {epoch} TRAIN seq_usage", train_seq_stats):
                    logging.info(line)
                for line in self._format_seq_block(f"Epoch {epoch} VALID seq_usage", valid_seq_stats):
                    logging.info(line)
                if self.writer:
                    self.writer.add_scalar('Throughput/train_samples_per_sec',
                                           train_thru, total_step)
                    if train_seq_stats is not None:
                        for d_idx, domain in enumerate(self._seq_domains_list):
                            self.writer.add_scalar(
                                f'SeqUsage/train_mean_len/{domain}',
                                train_seq_stats['mean_len'][d_idx], total_step)
                            self.writer.add_scalar(
                                f'SeqUsage/train_topk_sat/{domain}',
                                train_seq_stats['topk_sat_rate'][d_idx], total_step)
                            self.writer.add_scalar(
                                f'SeqUsage/train_trunc/{domain}',
                                train_seq_stats['trunc_rate'][d_idx], total_step)

                if train_dense_robust_stats is not None:
                    self._dense_robust_train_history.append({
                        'epoch': epoch, **train_dense_robust_stats})
                if valid_dense_robust_stats is not None:
                    self._dense_robust_valid_history.append({
                        'epoch': epoch, **valid_dense_robust_stats})
                for line in self._format_dense_robust_block(
                    f"Epoch {epoch} TRAIN DenseRobust", train_dense_robust_stats):
                    logging.info(line)
                for line in self._format_dense_robust_block(
                    f"Epoch {epoch} VALID DenseRobust", valid_dense_robust_stats):
                    logging.info(line)
                if self.writer and train_dense_robust_stats is not None:
                    for i, (fid, _offset, _length) in enumerate(self._dense_robust_blocks):
                        self.writer.add_scalar(
                            f'DenseRobust/train_max/fid_{fid}',
                            train_dense_robust_stats['max'][i], total_step)
                        self.writer.add_scalar(
                            f'DenseRobust/train_nonzero/fid_{fid}',
                            train_dense_robust_stats['nonzero_rate'][i], total_step)

            # Phase 0: effective rank diagnostics (main process only — SVD is
            # not free, and the result is identical across DDP ranks).
            if is_main_process() and not self.full_train:
                try:
                    eranks = self.compute_effective_ranks()
                    erank_str = ", ".join(f"{k}={v:.2f}" for k, v in eranks.items())
                    logging.info(f"Epoch {epoch} ERank | {erank_str}")
                    if self.writer:
                        for k, v in eranks.items():
                            self.writer.add_scalar(k, v, total_step)
                except Exception as e:
                    logging.warning(f"compute_effective_ranks failed: {e}")
                self.model.train()

            if self.full_train:
                if epoch == self.num_epochs:
                    self._save_full_train_checkpoint(total_step)
                    break
            else:
                self._handle_validation_result(total_step, val_auc, val_logloss)

                should_stop = self._broadcast_early_stop()
                if should_stop:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

            # Reinitialize high-cardinality sparse params (cold restart).
            if (
                epoch >= self.reinit_sparse_after_epoch
                and self.sparse_optimizer is not None
                and (not self.full_train or epoch < self.num_epochs)
            ):
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.raw_model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                # DDP: reinit uses random init, broadcast from rank 0 to sync all ranks.
                if is_ddp():
                    for p in self.raw_model.parameters():
                        dist.broadcast(p.data, src=0)
                sparse_params = self.raw_model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

                # V14: resync EMA shadow for re-initialized parameters so the
                # smoothed weights don't drag freshly-init'd params back toward
                # their pre-init values.
                if self.ema is not None:
                    self.ema.resync(self.raw_model, reinit_ptrs)

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ModelInput NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_gap_buckets: Dict[str, torch.Tensor] = {}
        seq_session_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_gap_buckets[domain] = device_batch.get(
                f'{domain}_gap_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_session_buckets[domain] = device_batch.get(
                f'{domain}_session_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            seq_gap_buckets=seq_gap_buckets,
            seq_session_buckets=seq_session_buckets,
        )

    def _sample_weights(self, device_batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Return per-row training weights normalized to mean 1 within batch."""
        if self.sample_weight_mode == 'none':
            return None
        ts = device_batch.get('timestamp')
        if not isinstance(ts, torch.Tensor):
            return None

        ts_f = ts.float()
        if self.sample_weight_mode == 'recency':
            if self.sample_weight_recency_ref_ts > 0:
                ref_ts = torch.full_like(ts_f, float(self.sample_weight_recency_ref_ts))
            else:
                ref_ts = ts_f.max().expand_as(ts_f)
            half_life = max(self.sample_weight_recency_half_life_days, 1e-6)
            age_days = torch.clamp((ref_ts - ts_f) / SECONDS_PER_DAY, min=0.0)
            raw = 1.0 + max(self.sample_weight_recency_max - 1.0, 0.0) * torch.exp(
                -age_days / half_life)
        else:
            ts_i = ts.long() + UTC8_OFFSET_SECONDS
            hour = torch.remainder(
                torch.div(ts_i, SECONDS_PER_HOUR, rounding_mode='floor'), 24)
            weekday = torch.remainder(
                torch.div(ts_i, SECONDS_PER_DAY, rounding_mode='floor') + 3, 7)
            raw = torch.ones_like(ts_f)
            if self.sample_weight_montue_boost > 0:
                raw = torch.where(
                    weekday <= 1,
                    raw * float(self.sample_weight_montue_boost),
                    raw,
                )
            if self.sample_weight_mode == 'montue_hour' and self.sample_weight_hour_strength > 0:
                hour_mult = self._test_hour_mult_cpu.to(
                    device=ts.device, dtype=ts_f.dtype)
                raw = raw * torch.pow(
                    hour_mult[hour],
                    float(self.sample_weight_hour_strength),
                )

        lo = min(self.sample_weight_min, self.sample_weight_max)
        hi = max(self.sample_weight_min, self.sample_weight_max)
        raw = torch.nan_to_num(raw, nan=1.0, posinf=hi, neginf=lo)
        raw = torch.clamp(raw, min=lo, max=hi)
        return raw / raw.mean().clamp_min(1e-6)

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        # V14: label smoothing for BCE/focal — pulls 0/1 targets toward 0.5
        # by `label_smoothing` magnitude. label = (1-ls)*y + 0.5*ls
        if self.label_smoothing > 0:
            label = label * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # V9.1: update target_match running stats before forward.
        if self._tm_train_acc_cur is not None and 'user_dense_feats' in device_batch:
            self._tm_update(self._tm_train_acc_cur, device_batch['user_dense_feats'])
        # V14: update seq-usage running stats before forward.
        if self._seq_train_acc_cur is not None:
            self._seq_update(self._seq_train_acc_cur, device_batch)
        if self._dense_robust_train_acc_cur is not None:
            self._dense_robust_update(self._dense_robust_train_acc_cur, device_batch)

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            sample_weights = self._sample_weights(device_batch)
            if self.loss_type == 'focal':
                loss_vec = sigmoid_focal_loss(
                    logits, label, alpha=self.focal_alpha,
                    gamma=self.focal_gamma, reduction='none')
            else:
                loss_vec = F.binary_cross_entropy_with_logits(
                    logits, label, reduction='none')
            if sample_weights is not None:
                loss = (loss_vec * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
            else:
                loss = loss_vec.mean()

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.unscale_(self.sparse_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
            self.scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.step(self.sparse_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        # V14: step LR scheduler + update EMA after each successful step.
        if self.dense_scheduler is not None:
            self.dense_scheduler.step()
        if self.ema is not None:
            self.ema.update(self.raw_model)

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation and return (AUC, logloss).

        In DDP mode, gathers results from all ranks and computes metrics on rank 0.
        Other ranks receive (0.0, 0.0).

        V14: if EMA is enabled, swap in shadow (EMA) weights for the duration of
        evaluation, then restore training weights.
        """
        if self.valid_loader is None:
            raise RuntimeError("Validation is disabled in full_train mode")
        if self.ema is not None:
            self.ema.apply_shadow(self.raw_model)
        try:
            return self._evaluate_inner(epoch)
        finally:
            if self.ema is not None:
                self.ema.restore(self.raw_model)

    def _evaluate_inner(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        if is_main_process():
            logging.info("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        # V9.1: reset per-evaluation target_match accumulator.
        self._tm_valid_acc_cur = self._tm_zero_acc()
        # V14: reset per-evaluation seq-usage accumulator.
        self._seq_valid_acc_cur = self._seq_zero_acc()
        # V14_DenseRobust: reset per-evaluation transformed dense monitor.
        self._dense_robust_valid_acc_cur = self._dense_robust_zero_acc()

        num_batches = self.valid_loader.dataset.estimated_num_batches()
        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            eval_t0 = time.time()
            for step, batch in enumerate(self.valid_loader):
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                if is_main_process() and (step + 1) % 100 == 0:
                    elapsed = time.time() - eval_t0
                    speed = (step + 1) / elapsed
                    eta = (num_batches - step - 1) / speed if speed > 0 else 0
                    print(f"Eval | step {step+1}/{num_batches} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # DDP: gather eval results from all ranks to rank 0.
        if is_ddp():
            all_logits, all_labels = self._gather_eval_results(all_logits, all_labels)

        # Only rank 0 computes metrics; other ranks return placeholder.
        if not is_main_process():
            return 0.0, 0.0

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _gather_eval_results(
        self,
        local_logits: torch.Tensor,
        local_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDP: gather eval results from all ranks to rank 0.

        Uses pad + all_gather to handle variable-length data from IterableDataset.
        """
        device = torch.device(self.device)
        world_size = dist.get_world_size()

        # 1. Gather each rank's sample count.
        local_size = torch.tensor([local_logits.shape[0]], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        sizes = [int(s.item()) for s in size_list]
        max_size = max(sizes)

        # 2. Pad to uniform length and all_gather.
        def _pad_and_gather(tensor, max_sz):
            padded = torch.zeros(max_sz, dtype=tensor.dtype, device=device)
            padded[:tensor.shape[0]] = tensor.to(device)
            gathered = [torch.zeros(max_sz, dtype=tensor.dtype, device=device) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
            parts = [g[:sz].cpu() for g, sz in zip(gathered, sizes)]
            return torch.cat(parts, dim=0)

        all_logits = _pad_and_gather(local_logits, max_size)
        all_labels = _pad_and_gather(local_labels, max_size)

        return all_logits, all_labels

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return (logits, labels)."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        # V9.1: update target_match valid running stats before forward.
        if self._tm_valid_acc_cur is not None and 'user_dense_feats' in device_batch:
            self._tm_update(self._tm_valid_acc_cur, device_batch['user_dense_feats'])
        # V14: update seq-usage valid running stats before forward.
        if self._seq_valid_acc_cur is not None:
            self._seq_update(self._seq_valid_acc_cur, device_batch)
        if self._dense_robust_valid_acc_cur is not None:
            self._dense_robust_update(self._dense_robust_valid_acc_cur, device_batch)

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits, _ = self.raw_model.predict(model_input)
        logits = logits.squeeze(-1)

        return logits, label

    # ── Phase 0: Effective rank monitoring ──
    # Reports erank at four points so we can locate where rank collapses:
    #   erank/ns_tokens        — NS tokenizer output (pre-block)
    #   erank/block_out_{i}    — query concat after block i (post token mixer)
    #   erank/query_concat     — final query concat just before output_proj
    #   erank/output           — post output_proj, fed to classifier
    # Uses up to _ERANK_COLLECT_BATCHES validation batches, capped at
    # _ERANK_MAX_ROWS rows.

    _ERANK_COLLECT_BATCHES = 10
    _ERANK_MAX_ROWS = 4096

    @torch.no_grad()
    def compute_effective_ranks(self) -> Dict[str, float]:
        """Sample validation batches and report effective rank of key reps."""
        if self.valid_loader is None:
            raise RuntimeError("ERank validation probe is disabled in full_train mode")
        was_training = self.model.training
        self.model.eval()
        raw = self.raw_model

        ns_parts = []
        query_concat_parts = []
        output_parts = []
        block_parts: Dict[int, list] = {}
        collected = 0

        for step, batch in enumerate(self.valid_loader):
            if step >= self._ERANK_COLLECT_BATCHES:
                break
            device_batch = self._batch_to_device(batch)
            model_input = self._make_model_input(device_batch)

            with self._autocast_context():
                # Replicate forward() NS construction to capture ns_tokens.
                user_ns = raw.user_ns_tokenizer(model_input.user_int_feats)
                item_ns = raw.item_ns_tokenizer(model_input.item_int_feats)
                ns_list = [user_ns]
                if raw.has_user_dense:
                    ns_list.append(raw.user_dense_tokenizer(model_input.user_dense_feats))
                ns_list.append(item_ns)
                if raw.has_item_dense:
                    ns_list.append(raw.item_dense_tokenizer(model_input.item_dense_feats))
                ns_tokens = torch.cat(ns_list, dim=1)

                seq_tokens_list, seq_masks_list = [], []
                for domain in raw.seq_domains:
                    tokens = raw._embed_seq_domain(
                        model_input.seq_data[domain],
                        raw._seq_embs[domain], raw._seq_proj[domain],
                        raw._seq_is_id[domain], raw._seq_emb_index[domain],
                        model_input.seq_time_buckets[domain])
                    seq_tokens_list.append(tokens)
                    seq_masks_list.append(raw._make_padding_mask(
                        model_input.seq_lens[domain],
                        model_input.seq_data[domain].shape[2]))

                q_tokens_list = raw.query_generator(
                    ns_tokens, seq_tokens_list, seq_masks_list)

                output, intermediates = raw._run_multi_seq_blocks(
                    q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
                    apply_dropout=False, return_intermediates=True)

            ns_parts.append(ns_tokens.float().cpu().flatten(1))
            query_concat_parts.append(intermediates['query_concat'].float().cpu())
            output_parts.append(output.float().cpu())
            for i, bo in enumerate(intermediates['block_outs']):
                block_parts.setdefault(i, []).append(bo.float().cpu())

            collected += output.shape[0]
            if collected >= self._ERANK_MAX_ROWS:
                break

        results: Dict[str, float] = {}
        if ns_parts:
            ns_mat = torch.cat(ns_parts, dim=0)[:self._ERANK_MAX_ROWS]
            results['erank/ns_tokens'] = effective_rank(ns_mat)
        if query_concat_parts:
            qc_mat = torch.cat(query_concat_parts, dim=0)[:self._ERANK_MAX_ROWS]
            results['erank/query_concat'] = effective_rank(qc_mat)
        if output_parts:
            out_mat = torch.cat(output_parts, dim=0)[:self._ERANK_MAX_ROWS]
            results['erank/output'] = effective_rank(out_mat)
        for i, parts in block_parts.items():
            mat = torch.cat(parts, dim=0)[:self._ERANK_MAX_ROWS]
            results[f'erank/block_out_{i}'] = effective_rank(mat)

        if was_training:
            self.model.train()
        return results
