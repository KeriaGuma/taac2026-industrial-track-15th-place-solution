"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


CATEGORY_PADDING_ID = 0
CATEGORY_MISSING_ID = 1
CATEGORY_OOB_LOW_ID = 2
CATEGORY_OOB_HIGH_ID = 3
CATEGORY_RESERVED_OFFSET = 3
CATEGORY_RESERVED_VOCAB = 4


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


# ─────────────────────────── V1 Synthetic Features ───────────────────────────

ABS_TIME_USER_INT_COLS = [
    [1000001, 25, 1],   # UTC+8 hour, values 1..24
    [1000002, 8, 1],    # UTC+8 weekday, values 1..7, Monday=1
    [1000003, 169, 1],  # hour-weekday cross, values 1..168
]

SEQ_ACTIVITY_DENSE_COL = [2000001, 16]
LONG_HISTORY_DENSE_COL = [2000002, 32]
SEQ_TRUNC_DENSE_COL = [2000003, 16]
MISSING_INDICATOR_DENSE_COL = [2000004, 12]
TARGET_MATCH_DENSE_COL = [2000005, 20]

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
UTC8_OFFSET_SECONDS = 8 * SECONDS_PER_HOUR
HISTORY_WINDOWS = (3600, 21600, 86400, 604800, 2592000)
SESSION_GAP_SECONDS = 1800


def parse_seq_fid_map(value: Optional[str]) -> Dict[str, List[int]]:
    if not value:
        return {}
    result: Dict[str, List[int]] = {}
    for chunk in str(value).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        domain, raw_fids = chunk.split(":", 1)
        fids = [int(v.strip()) for v in raw_fids.split(",") if v.strip()]
        if fids:
            result[domain.strip()] = sorted(set(fids))
    return result


def parse_int_csv(value: Optional[str]) -> List[int]:
    if not value:
        return []
    return sorted({
        int(v.strip())
        for v in str(value).replace(";", ",").split(",")
        if v.strip()
    })


def build_abs_time_user_int(timestamps: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.int64) + UTC8_OFFSET_SECONDS
    hours = (np.floor_divide(ts, SECONDS_PER_HOUR) % 24) + 1
    days = np.floor_divide(ts, SECONDS_PER_DAY)
    weekdays = ((days + 3) % 7) + 1
    cross = (hours - 1) * 7 + weekdays
    return np.stack([hours, weekdays, cross], axis=1).astype(np.int64, copy=False)


def _safe_log1p(value: float) -> float:
    return float(np.log1p(max(0.0, value)))


def _window_counts(
    deltas: np.ndarray,
    windows: Sequence[int] = HISTORY_WINDOWS,
) -> List[float]:
    if deltas.size == 0:
        return [0.0] * len(windows)
    return [_safe_log1p(float(np.count_nonzero(deltas <= w))) for w in windows]


def _sequence_time_stats_for_row(
    current_ts: int,
    values: np.ndarray,
    start: int,
    end: int,
) -> Tuple[float, float, float, List[float]]:
    if end <= start:
        return 0.0, 0.0, 0.0, [0.0] * len(HISTORY_WINDOWS)
    raw_ts = np.asarray(values[start:end], dtype=np.int64)
    raw_ts = raw_ts[raw_ts > 0]
    if raw_ts.size == 0:
        return 0.0, 0.0, 0.0, [0.0] * len(HISTORY_WINDOWS)
    deltas = np.maximum(int(current_ts) - raw_ts, 0)
    nearest = float(deltas.min())
    farthest = float(deltas.max())
    span = float(max(0, raw_ts.max() - raw_ts.min()))
    return _safe_log1p(nearest), _safe_log1p(farthest), _safe_log1p(span), _window_counts(deltas)


def fill_sequence_activity_dense(
    out: np.ndarray,
    offset: int,
    domain_order: Sequence[str],
    raw_lengths: Mapping[str, np.ndarray],
    used_lengths: Mapping[str, np.ndarray],
    max_lens: Mapping[str, int],
) -> None:
    for d_idx, domain in enumerate(domain_order):
        base = offset + d_idx * 4
        raw_len = raw_lengths[domain].astype(np.float32, copy=False)
        used_len = used_lengths[domain].astype(np.float32, copy=False)
        max_len = float(max_lens[domain])
        out[:, base + 0] = np.log1p(raw_len)
        out[:, base + 1] = np.log1p(used_len)
        out[:, base + 2] = (raw_len <= 0).astype(np.float32)
        out[:, base + 3] = np.minimum(raw_len, max_len) / max(max_len, 1.0)


def fill_truncation_dense(
    out: np.ndarray,
    offset: int,
    domain_order: Sequence[str],
    raw_lengths: Mapping[str, np.ndarray],
    max_lens: Mapping[str, int],
) -> None:
    for d_idx, domain in enumerate(domain_order):
        base = offset + d_idx * 4
        raw_len = raw_lengths[domain].astype(np.float32, copy=False)
        max_len = float(max_lens[domain])
        overflow = np.maximum(raw_len - max_len, 0.0)
        out[:, base + 0] = np.log1p(raw_len)
        out[:, base + 1] = _safe_log1p(max_len)
        out[:, base + 2] = np.log1p(overflow)
        out[:, base + 3] = overflow / np.maximum(raw_len, 1.0)


def fill_long_history_dense(
    out: np.ndarray,
    offset: int,
    domain_order: Sequence[str],
    timestamps: np.ndarray,
    ts_arrays: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
) -> None:
    for d_idx, domain in enumerate(domain_order):
        base = offset + d_idx * 8
        ts_info = ts_arrays.get(domain)
        if ts_info is None:
            continue
        offsets, values = ts_info
        for i in range(batch_size):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            nearest, farthest, span, counts = _sequence_time_stats_for_row(
                int(timestamps[i]), values, start, end)
            out[i, base + 0] = nearest
            out[i, base + 1] = farthest
            out[i, base + 2] = span
            out[i, base + 3:base + 8] = counts


def fill_missing_indicator_dense(
    out: np.ndarray,
    offset: int,
    user_int: np.ndarray,
    item_int: np.ndarray,
    user_dense: np.ndarray,
    item_dense: np.ndarray,
    raw_lengths: Mapping[str, np.ndarray],
    domain_order: Sequence[str],
) -> None:
    out[:, offset + 0] = (user_int <= 0).mean(axis=1) if user_int.shape[1] else 0.0
    out[:, offset + 1] = (item_int <= 0).mean(axis=1) if item_int.shape[1] else 0.0
    out[:, offset + 2] = (user_dense == 0).mean(axis=1) if user_dense.shape[1] else 0.0
    out[:, offset + 3] = (item_dense == 0).mean(axis=1) if item_dense.shape[1] else 0.0
    out[:, offset + 4] = np.isfinite(user_dense).mean(axis=1) if user_dense.shape[1] else 1.0
    out[:, offset + 5] = np.isfinite(item_dense).mean(axis=1) if item_dense.shape[1] else 1.0
    for idx, domain in enumerate(domain_order[:4]):
        out[:, offset + 6 + idx] = (raw_lengths[domain] <= 0).astype(np.float32)
    total_raw = np.zeros(out.shape[0], dtype=np.float32)
    non_empty = np.zeros(out.shape[0], dtype=np.float32)
    for domain in domain_order:
        length = raw_lengths[domain].astype(np.float32, copy=False)
        total_raw += length
        non_empty += (length > 0).astype(np.float32)
    out[:, offset + 10] = np.log1p(total_raw)
    out[:, offset + 11] = non_empty / max(float(len(domain_order)), 1.0)


def fill_target_match_dense(
    out: np.ndarray,
    offset: int,
    domain_order: Sequence[str],
    target_items: Optional[np.ndarray],
    match_arrays: Mapping[str, List[Tuple[np.ndarray, np.ndarray]]],
    ts_arrays: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    timestamps: np.ndarray,
    batch_size: int,
) -> None:
    if target_items is None:
        return
    for d_idx, domain in enumerate(domain_order):
        base = offset + d_idx * 5
        arrays = match_arrays.get(domain, [])
        if not arrays:
            continue
        ts_info = ts_arrays.get(domain)
        ts_offsets, ts_values = ts_info if ts_info is not None else (None, None)
        for i in range(batch_size):
            target = target_items[i]
            if target <= 0:
                continue
            positions: List[int] = []
            for offsets, values in arrays:
                start = int(offsets[i])
                end = int(offsets[i + 1])
                if end <= start:
                    continue
                vals = np.asarray(values[start:end], dtype=np.int64)
                hit = np.flatnonzero(vals == target)
                if hit.size:
                    positions.extend(int(v) for v in hit)
            if not positions:
                continue
            positions_arr = np.asarray(positions, dtype=np.int64)
            out[i, base + 0] = 1.0
            out[i, base + 1] = _safe_log1p(float(positions_arr.size))
            out[i, base + 2] = 1.0 if np.any(positions_arr < 20) else 0.0
            out[i, base + 3] = 1.0 if np.any(positions_arr < 100) else 0.0
            if ts_offsets is not None and ts_values is not None:
                start = int(ts_offsets[i])
                end = int(ts_offsets[i + 1])
                valid_pos = positions_arr[positions_arr < max(0, end - start)]
                if valid_pos.size:
                    hit_ts = np.asarray(ts_values[start:end], dtype=np.int64)[valid_pos]
                    hit_ts = hit_ts[hit_ts > 0]
                    if hit_ts.size:
                        gap = np.maximum(int(timestamps[i]) - hit_ts, 0).min()
                        out[i, base + 4] = _safe_log1p(float(gap))


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        seed: int = 42,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        timestamp_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        use_abs_time_user_int: bool = False,
        use_sequence_summary_dense: bool = False,
        use_seq_trunc_dense: bool = False,
        use_missing_indicator_dense: bool = False,
        use_target_match_dense: bool = False,
        use_inter_event_gap_session: bool = False,
        target_match_seq_fids: str = "",
        dense_robust_user_fids: str = "",
        dense_robust_clip: float = 16.0,
        dense_robust_scale: bool = True,
        use_reserved_categorical_ids: bool = False,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            seed: random seed for deterministic shuffle in _flush_buffer.
            ddp_rank: DDP rank (passed from the main process; worker subprocesses inherit).
            ddp_world_size: DDP world_size.
            timestamp_range: optional ``(lo, hi)`` to filter rows by timestamp.
                Only rows with ``lo < ts <= hi`` are kept. Use None for open ends.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self._seed = seed
        self._flush_count = 0
        self._ts_range = timestamp_range
        self.use_abs_time_user_int = bool(use_abs_time_user_int)
        self.use_sequence_summary_dense = bool(use_sequence_summary_dense)
        self.use_seq_trunc_dense = bool(use_seq_trunc_dense)
        self.use_missing_indicator_dense = bool(use_missing_indicator_dense)
        self.use_target_match_dense = bool(use_target_match_dense)
        self.use_inter_event_gap_session = bool(use_inter_event_gap_session)
        self.target_match_fids = parse_seq_fid_map(target_match_seq_fids)
        self.dense_robust_user_fids = parse_int_csv(dense_robust_user_fids)
        self.dense_robust_clip = float(dense_robust_clip)
        self.dense_robust_scale = bool(dense_robust_scale)
        self.use_reserved_categorical_ids = bool(use_reserved_categorical_ids)
        self._dense_robust_user_blocks: List[Tuple[int, int, int]] = []
        if self.use_reserved_categorical_ids:
            logging.info(
                "V18_ReservedCatIds enabled: "
                "0=padding, 1=missing/null, 2=oob_low/non-positive, "
                "3=oob_high, 4+=raw_id+3; model vocab sizes are expanded."
            )
        if self.use_inter_event_gap_session:
            logging.info(
                "V16_InterEventGapSession enabled: per-token gap_bucket + "
                "session_bucket, session_gap_seconds=%d",
                SESSION_GAP_SECONDS,
            )
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        # DDP rank-level sharding: greedy allocation by row count to balance across ranks.
        if ddp_world_size > 1:
            sorted_rgs = sorted(self._rg_list, key=lambda x: x[2], reverse=True)
            rank_buckets = [[] for _ in range(ddp_world_size)]
            rank_rows = [0] * ddp_world_size
            for rg in sorted_rgs:
                min_rank = min(range(ddp_world_size), key=lambda r: rank_rows[r])
                rank_buckets[min_rank].append(rg)
                rank_rows[min_rank] += rg[2]
            self._rg_list = rank_buckets[ddp_rank]
            logging.info(f"DDP shard: rank={ddp_rank}, world_size={ddp_world_size}, "
                         f"row_groups={len(self._rg_list)}, rows={rank_rows[ddp_rank]} "
                         f"(all ranks: {rank_rows})")

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_item_dense = np.zeros((B, self.item_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            if ci is not None:
                self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            if ci is not None:
                self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            if ci is not None:
                self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        self._item_dense_plan = []
        offset = 0
        for fid, dim in self._item_dense_cols:
            ci = self._col_idx.get(f'item_dense_feats_{fid}')
            if ci is not None:
                self._item_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([self._model_vocab_size(vs)] * dim)
        self._raw_user_int_total_dim = self.user_int_schema.total_dim
        self._synthetic_user_int_offsets: Dict[int, Tuple[int, int]] = {}
        if self.use_abs_time_user_int:
            for fid, vs, dim in ABS_TIME_USER_INT_COLS:
                self._synthetic_user_int_offsets[fid] = (
                    self.user_int_schema.total_dim, dim)
                self._user_int_cols.append([fid, vs, dim])
                self.user_int_schema.add(fid, dim)
                self.user_int_vocab_sizes.extend([self._model_vocab_size(vs)] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([self._model_vocab_size(vs)] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        # Override dims for features whose effective length < schema max_len.
        _USER_DENSE_DIM_OVERRIDE = {
            130: 259,  # effective dim=259, tail is unrelated stats
        }
        self._user_dense_cols: List[List[int]] = [
            [fid, _USER_DENSE_DIM_OVERRIDE.get(fid, dim)]
            for fid, dim in raw['user_dense']
        ]
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)
        self._dense_robust_user_blocks = []
        if self.dense_robust_user_fids:
            for fid, offset, length in self.user_dense_schema.entries:
                if fid in self.dense_robust_user_fids:
                    self._dense_robust_user_blocks.append(
                        (int(fid), int(offset), int(length)))
            missing = sorted(set(self.dense_robust_user_fids) - {
                fid for fid, _, _ in self._dense_robust_user_blocks
            })
            logging.info(
                "V14_DenseRobust enabled: "
                f"requested_user_fids={self.dense_robust_user_fids}, "
                f"active_blocks={self._dense_robust_user_blocks}, "
                f"missing={missing}, log1p=True, clip={self.dense_robust_clip}, "
                f"scale={self.dense_robust_scale}"
            )
        self._raw_user_dense_total_dim = self.user_dense_schema.total_dim
        self._synthetic_user_dense_offsets: Dict[int, Tuple[int, int]] = {}
        for enabled, col in [
            (self.use_sequence_summary_dense, SEQ_ACTIVITY_DENSE_COL),
            (self.use_sequence_summary_dense, LONG_HISTORY_DENSE_COL),
            (self.use_seq_trunc_dense, SEQ_TRUNC_DENSE_COL),
            (self.use_missing_indicator_dense, MISSING_INDICATOR_DENSE_COL),
            (self.use_target_match_dense, TARGET_MATCH_DENSE_COL),
        ]:
            if not enabled:
                continue
            fid, dim = col
            self._synthetic_user_dense_offsets[fid] = (
                self.user_dense_schema.total_dim, dim)
            self._user_dense_cols.append([fid, dim])
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense: [[fid, dim], ...] ----
        self._item_dense_cols: List[List[int]] = raw.get('item_dense', [])
        self.item_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._item_dense_cols:
            self.item_dense_schema.add(fid, dim)
        self._raw_item_dense_total_dim = self.item_dense_schema.total_dim

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self._model_vocab_size(self.seq_vocab_sizes[domain][fid])
                for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def estimated_num_batches(self) -> int:
        """Estimate batch count for logging/progress display only."""
        return (self.num_rows + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        self._flush_count = 0  # Reset each epoch

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if batch_dict is None:
                    continue
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + self._flush_count)
            self._flush_count += 1
            rand_idx = torch.randperm(total_rows, generator=g)
        else:
            rand_idx = torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _apply_user_dense_robust(self, user_dense: np.ndarray) -> None:
        if not self._dense_robust_user_blocks:
            return
        clip = max(float(self.dense_robust_clip), 0.0)
        for _fid, offset, length in self._dense_robust_user_blocks:
            block = user_dense[:, offset:offset + length]
            np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(block, 0.0, out=block)
            np.log1p(block, out=block)
            if clip > 0:
                np.minimum(block, clip, out=block)
                if self.dense_robust_scale:
                    block /= clip

    def _model_vocab_size(self, raw_vocab_size: int) -> int:
        raw_vocab_size = int(raw_vocab_size)
        if not self.use_reserved_categorical_ids:
            return raw_vocab_size
        if raw_vocab_size <= 0:
            return CATEGORY_RESERVED_VOCAB
        return raw_vocab_size + CATEGORY_RESERVED_OFFSET

    def _encode_reserved_ids(
        self,
        arr: "npt.NDArray[np.int64]",
        raw_vocab_size: int,
        valid_mask: Optional["npt.NDArray[np.bool_]"] = None,
    ) -> "npt.NDArray[np.int64]":
        raw = np.asarray(arr, dtype=np.int64)
        out = np.zeros(raw.shape, dtype=np.int64)
        if valid_mask is None:
            valid = np.ones(raw.shape, dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            out[~valid] = CATEGORY_MISSING_ID

        out[valid & (raw <= 0)] = CATEGORY_OOB_LOW_ID
        raw_vocab_size = int(raw_vocab_size)
        if raw_vocab_size <= 0:
            out[valid & (raw > 0)] = CATEGORY_OOB_HIGH_ID
            return out

        out[valid & (raw >= raw_vocab_size)] = CATEGORY_OOB_HIGH_ID
        ok = valid & (raw > 0) & (raw < raw_vocab_size)
        out[ok] = raw[ok] + CATEGORY_RESERVED_OFFSET
        return out

    def _encode_scalar_reserved_column(
        self,
        col: "pa.Array",
        raw_vocab_size: int,
    ) -> "npt.NDArray[np.int64]":
        arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        valid = np.logical_not(
            col.is_null().to_numpy(zero_copy_only=False).astype(bool)
        )
        return self._encode_reserved_ids(arr, raw_vocab_size, valid)

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
        raw_vocab_size: Optional[int] = None,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values_array = arrow_col.values
        values = values_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        value_valid = np.logical_not(
            values_array.is_null().to_numpy(zero_copy_only=False).astype(bool)
        )

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            raw_slice = values[start:start + use_len]
            if self.use_reserved_categorical_ids and raw_vocab_size is not None:
                valid_slice = value_valid[start:start + use_len]
                padded[i, :use_len] = self._encode_reserved_ids(
                    raw_slice, int(raw_vocab_size), valid_slice)
            else:
                padded[i, :use_len] = raw_slice
            lengths[i] = use_len

        if not (self.use_reserved_categorical_ids and raw_vocab_size is not None):
            padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _collect_sequence_context(
        self,
        batch: "pa.RecordBatch",
        B: int,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, Tuple[np.ndarray, np.ndarray]],
        Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    ]:
        raw_lengths: Dict[str, np.ndarray] = {}
        used_lengths: Dict[str, np.ndarray] = {}
        ts_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        match_arrays: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for domain in self.seq_domains:
            side_plan, ts_ci = self._seq_plan[domain]
            length_ci = ts_ci if ts_ci is not None else (
                side_plan[0][0] if side_plan else None)
            if length_ci is None:
                lengths = np.zeros(B, dtype=np.int64)
            else:
                col = batch.column(length_ci)
                offsets = col.offsets.to_numpy()
                lengths = np.asarray(
                    [max(0, int(offsets[i + 1]) - int(offsets[i]))
                     for i in range(B)],
                    dtype=np.int64,
                )
            raw_lengths[domain] = lengths
            used_lengths[domain] = np.minimum(lengths, self._seq_maxlen[domain])

            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_arrays[domain] = (
                    ts_col.offsets.to_numpy(), ts_col.values.to_numpy())

            requested_fids = set(self.target_match_fids.get(domain, []))
            if requested_fids:
                prefix = self._seq_prefix[domain]
                arrays = []
                for fid in requested_fids:
                    ci = self._col_idx.get(f'{prefix}_{fid}')
                    if ci is None:
                        continue
                    col = batch.column(ci)
                    arrays.append((col.offsets.to_numpy(), col.values.to_numpy()))
                if arrays:
                    match_arrays[domain] = arrays

        return raw_lengths, used_lengths, ts_arrays, match_arrays

    def _convert_batch(self, batch: "pa.RecordBatch") -> Optional[Dict[str, Any]]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""

        # ---- timestamp range filter ----
        if self._ts_range is not None:
            lo, hi = self._ts_range
            ts_col = batch.column(self._col_idx['timestamp'])
            if lo is not None and hi is not None:
                mask = pc.and_(pc.greater(ts_col, lo), pc.less_equal(ts_col, hi))
            elif hi is not None:
                mask = pc.less_equal(ts_col, hi)
            else:
                mask = pc.greater(ts_col, lo)
            batch = batch.filter(mask)
            if batch.num_rows == 0:
                return None

        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()
        seq_raw_lengths, seq_used_lengths, seq_ts_arrays, seq_match_arrays = (
            self._collect_sequence_context(batch, B))

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                if self.use_reserved_categorical_ids:
                    arr = self._encode_scalar_reserved_column(col, vs)
                else:
                    arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                    arr[arr <= 0] = 0
                    if vs > 0:
                        self._record_oob('user_int', ci, arr, vs)
                    else:
                        arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(
                    col, dim, B,
                    raw_vocab_size=vs if self.use_reserved_categorical_ids else None,
                )
                if not self.use_reserved_categorical_ids:
                    if vs > 0:
                        self._record_oob('user_int', ci, padded, vs)
                    else:
                        padded[:] = 0
                user_int[:, offset:offset + dim] = padded
        if self.use_abs_time_user_int:
            abs_time = build_abs_time_user_int(timestamps)
            if self.use_reserved_categorical_ids:
                abs_time = abs_time + CATEGORY_RESERVED_OFFSET
            for j, (fid, _vs, dim) in enumerate(ABS_TIME_USER_INT_COLS):
                offset, _ = self._synthetic_user_int_offsets[fid]
                user_int[:, offset:offset + dim] = abs_time[:, j:j + dim]

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                if self.use_reserved_categorical_ids:
                    arr = self._encode_scalar_reserved_column(col, vs)
                else:
                    arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                    arr[arr <= 0] = 0
                    if vs > 0:
                        self._record_oob('item_int', ci, arr, vs)
                    else:
                        arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(
                    col, dim, B,
                    raw_vocab_size=vs if self.use_reserved_categorical_ids else None,
                )
                if not self.use_reserved_categorical_ids:
                    if vs > 0:
                        self._record_oob('item_int', ci, padded, vs)
                    else:
                        padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        # ---- item_dense ----
        if self.item_dense_schema.total_dim > 0:
            item_dense = self._buf_item_dense[:B]
            item_dense[:] = 0
            for ci, dim, offset in self._item_dense_plan:
                col = batch.column(ci)
                padded = self._pad_varlen_float_column(col, dim, B)
                item_dense[:, offset:offset + dim] = padded
            item_dense_tensor = torch.from_numpy(item_dense.copy())
        else:
            item_dense_tensor = torch.zeros(B, 0, dtype=torch.float32)
            item_dense = np.zeros((B, 0), dtype=np.float32)

        if self.use_sequence_summary_dense:
            offset, _ = self._synthetic_user_dense_offsets[
                SEQ_ACTIVITY_DENSE_COL[0]]
            fill_sequence_activity_dense(
                user_dense, offset, self.seq_domains, seq_raw_lengths,
                seq_used_lengths, self._seq_maxlen)
            offset, _ = self._synthetic_user_dense_offsets[
                LONG_HISTORY_DENSE_COL[0]]
            fill_long_history_dense(
                user_dense, offset, self.seq_domains, timestamps,
                seq_ts_arrays, B)
        if self.use_seq_trunc_dense:
            offset, _ = self._synthetic_user_dense_offsets[
                SEQ_TRUNC_DENSE_COL[0]]
            fill_truncation_dense(
                user_dense, offset, self.seq_domains, seq_raw_lengths,
                self._seq_maxlen)
        if self.use_missing_indicator_dense:
            offset, _ = self._synthetic_user_dense_offsets[
                MISSING_INDICATOR_DENSE_COL[0]]
            fill_missing_indicator_dense(
                user_dense, offset, user_int[:, :self._raw_user_int_total_dim], item_int,
                user_dense[:, :self._raw_user_dense_total_dim],
                item_dense[:, :self._raw_item_dense_total_dim],
                seq_raw_lengths, self.seq_domains)
        if self.use_target_match_dense:
            offset, _ = self._synthetic_user_dense_offsets[
                TARGET_MATCH_DENSE_COL[0]]
            target_items = None
            item_id_ci = self._col_idx.get('item_id')
            if item_id_ci is not None:
                target_items = (
                    batch.column(item_id_ci).fill_null(0)
                    .to_numpy(zero_copy_only=False).astype(np.int64)
                )
            fill_target_match_dense(
                user_dense, offset, self.seq_domains, target_items,
                seq_match_arrays, seq_ts_arrays, timestamps, B)

        self._apply_user_dense_robust(user_dense)

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': item_dense_tensor,
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }
        item_id_ci = self._col_idx.get('item_id')
        if item_id_ci is not None:
            item_ids = (
                batch.column(item_id_ci).fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int64)
            )
            result['item_id'] = torch.from_numpy(item_ids.copy())

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                values_array = col.values
                values = values_array.fill_null(0).to_numpy(
                    zero_copy_only=False).astype(np.int64)
                valid = np.logical_not(
                    values_array.is_null().to_numpy(
                        zero_copy_only=False).astype(bool)
                )
                col_data.append((col.offsets.to_numpy(), values, valid, vs, ci))

            for c, (offs, vals, valid_vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    raw_slice = vals[s:s + ul]
                    if self.use_reserved_categorical_ids:
                        valid_slice = valid_vals[s:s + ul]
                        out[i, c, :ul] = self._encode_reserved_ids(
                            raw_slice, vs, valid_slice)
                    else:
                        out[i, c, :ul] = raw_slice
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0 in the legacy path. The reserved-id path keeps
            # real non-positive events as CATEGORY_OOB_LOW_ID while padded
            # positions remain 0.
            if not self.use_reserved_categorical_ids:
                out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if not self.use_reserved_categorical_ids:
                    if vs > 0:
                        self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                    else:
                        slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            gap_bucket = None
            session_bucket = None
            if self.use_inter_event_gap_session:
                gap_bucket = np.zeros((B, max_len), dtype=np.int64)
                session_bucket = np.zeros((B, max_len), dtype=np.int64)
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

                if self.use_inter_event_gap_session:
                    valid = ts_padded > 0
                    gaps = np.zeros((B, max_len), dtype=np.int64)
                    if max_len > 1:
                        gaps[:, 1:] = np.maximum(
                            ts_padded[:, :-1] - ts_padded[:, 1:],
                            0,
                        )
                    raw_gap_buckets = np.clip(
                        np.searchsorted(BUCKET_BOUNDARIES, gaps.ravel()),
                        0, len(BUCKET_BOUNDARIES) - 1,
                    )
                    gap_bucket[:] = raw_gap_buckets.reshape(B, max_len) + 1
                    gap_bucket[~valid] = 0
                    gap_bucket[:, 0] = 0

                    session_bucket[valid] = 1
                    first_valid = valid[:, 0]
                    session_bucket[first_valid, 0] = 2
                    new_session = (gaps > SESSION_GAP_SECONDS) & valid
                    session_bucket[new_session] = 2

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            if self.use_inter_event_gap_session:
                result[f'{domain}_gap_bucket'] = torch.from_numpy(gap_bucket.copy())
                result[f'{domain}_session_bucket'] = torch.from_numpy(session_bucket.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    full_train: bool = False,
    num_workers: int = 16,
    prefetch_factor: int = 2,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    split_by_time: bool = False,
    **kwargs: Any,
) -> Tuple[DataLoader, Optional[DataLoader], PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    Split modes:
      - ``split_by_time=False`` (default): split by Row Group position.
        The last ``valid_ratio`` fraction of Row Groups becomes validation.
      - ``split_by_time=True``: split by timestamp. Scan the timestamp column
        to find a cutoff so that the last ``valid_ratio`` of rows (by time
        order) become validation.
      - ``full_train=True`` or row-group ``valid_ratio <= 0``: use every Row
        Group for training and return ``valid_loader=None``.

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    use_pin_memory = torch.cuda.is_available() and _env_flag("PIN_MEMORY", True)
    logging.info(f"DataLoader pin_memory={use_pin_memory}")
    feature_kwargs = {
        'use_abs_time_user_int': kwargs.get('use_abs_time_user_int', False),
        'use_sequence_summary_dense': kwargs.get('use_sequence_summary_dense', False),
        'use_seq_trunc_dense': kwargs.get('use_seq_trunc_dense', False),
        'use_missing_indicator_dense': kwargs.get('use_missing_indicator_dense', False),
        'use_target_match_dense': kwargs.get('use_target_match_dense', False),
        'use_reserved_categorical_ids': kwargs.get('use_reserved_categorical_ids', False),
        'use_inter_event_gap_session': kwargs.get('use_inter_event_gap_session', False),
        'target_match_seq_fids': kwargs.get('target_match_seq_fids', ''),
        'dense_robust_user_fids': kwargs.get('dense_robust_user_fids', ''),
        'dense_robust_clip': kwargs.get('dense_robust_clip', 16.0),
        'dense_robust_scale': kwargs.get('dense_robust_scale', True),
    }

    no_validation = bool(full_train) or (
        not split_by_time and float(valid_ratio) <= 0.0
    )
    if no_validation:
        if split_by_time:
            raise ValueError("full_train/no-validation mode requires split_by_time=False")

        train_rows = sum(r[2] for r in rg_info)
        logging.info(
            "Full-train split: using all %d Row Groups (%d rows) for training; "
            "validation is disabled.",
            total_rgs,
            train_rows,
        )

        train_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=shuffle_train,
            buffer_batches=buffer_batches,
            row_group_range=(0, total_rgs),
            clip_vocab=clip_vocab,
            seed=seed,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            **feature_kwargs,
        )

        _train_kw = {}
        if num_workers > 0:
            _train_kw['persistent_workers'] = True
            _train_kw['prefetch_factor'] = max(1, int(prefetch_factor))

        train_loader = DataLoader(
            train_dataset, batch_size=None,
            num_workers=num_workers, pin_memory=use_pin_memory, **_train_kw,
        )
        logging.info(
            "Parquet full-train: train=%d rows, valid=0 rows, "
            "batch_size=%d, buffer_batches=%d",
            train_rows,
            batch_size,
            buffer_batches,
        )
        return train_loader, None, train_dataset

    if split_by_time:
        # ---- Time-based split: find timestamp cutoff at (1 - valid_ratio) quantile ----
        from concurrent.futures import ThreadPoolExecutor

        def _read_ts(fpath):
            return pq.read_table(fpath, columns=['timestamp']).column('timestamp').to_numpy()

        with ThreadPoolExecutor(max_workers=32) as pool:
            all_ts = list(pool.map(_read_ts, pq_files))
        all_ts = np.concatenate(all_ts)
        total_rows = len(all_ts)

        cutoff = int(np.percentile(all_ts, (1.0 - valid_ratio) * 100))
        train_rows = int((all_ts <= cutoff).sum())
        valid_rows = total_rows - train_rows

        logging.info(f"Time-based split: cutoff_ts={cutoff}, "
                     f"train={train_rows} rows, valid={valid_rows} rows, "
                     f"total={total_rows}")
        del all_ts

        train_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=shuffle_train,
            buffer_batches=buffer_batches,
            clip_vocab=clip_vocab,
            seed=seed,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            timestamp_range=(None, cutoff),
            **feature_kwargs,
        )

        _train_kw = {}
        if num_workers > 0:
            _train_kw['persistent_workers'] = True
            _train_kw['prefetch_factor'] = max(1, int(prefetch_factor))

        train_loader = DataLoader(
            train_dataset, batch_size=None,
            num_workers=num_workers, pin_memory=use_pin_memory, **_train_kw,
        )

        valid_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            clip_vocab=clip_vocab,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            timestamp_range=(cutoff, None),
            **feature_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=None,
            num_workers=0, pin_memory=use_pin_memory,
        )

        logging.info(f"Parquet (time split): train={train_rows}, valid={valid_rows}, "
                     f"batch_size={batch_size}, buffer_batches={buffer_batches}")

        return train_loader, valid_loader, train_dataset

    # ---- Row Group position-based split (original behavior) ----
    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        seed=seed,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        **feature_kwargs,
    )

    use_pin_memory = torch.cuda.is_available() and _env_flag("PIN_MEMORY", True)
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = max(1, int(prefetch_factor))

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_pin_memory, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        **feature_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_pin_memory,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
