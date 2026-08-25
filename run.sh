#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIN_MEMORY="${PIN_MEMORY:-0}"

# ── GPU detection ──
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES// /}" ]]; then
    NGPUS=0
  else
    NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
  fi
elif command -v nvidia-smi &> /dev/null; then
  NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  NGPUS=0
fi
echo "Using ${NGPUS} GPU(s) (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}')"

# ── Resource-aware throughput knobs (identical to V9/V11/V12/V13) ──
CPU_CORES=1
if command -v getconf &> /dev/null; then
  CPU_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
elif command -v sysctl &> /dev/null; then
  CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
fi

GPU_MEM_MB=0
if [[ "${NGPUS}" -gt 0 ]] && command -v nvidia-smi &> /dev/null; then
  GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NR == 1 || $1 < min {min = $1} END {print min + 0}')
fi

if [[ "${NGPUS}" -gt 0 ]]; then
  WORKERS_PER_RANK=$(( CPU_CORES / NGPUS ))
else
  WORKERS_PER_RANK=$(( CPU_CORES / 2 ))
fi
if [[ "${WORKERS_PER_RANK}" -lt 2 ]]; then
  WORKERS_PER_RANK=2
fi

if [[ "${GPU_MEM_MB}" -ge 76000 ]]; then
  # V16_AsymSeqAHeavy widens D=114 and gives seq_a extra query tokens; keep headroom.
  AUTO_BATCH_SIZE=1024
  AUTO_PREFETCH_FACTOR=4
  MAX_WORKERS=16
elif [[ "${GPU_MEM_MB}" -ge 40000 ]]; then
  AUTO_BATCH_SIZE=768
  AUTO_PREFETCH_FACTOR=4
  MAX_WORKERS=12
elif [[ "${GPU_MEM_MB}" -ge 23000 ]]; then
  AUTO_BATCH_SIZE=512
  AUTO_PREFETCH_FACTOR=3
  MAX_WORKERS=8
elif [[ "${GPU_MEM_MB}" -ge 15000 ]]; then
  AUTO_BATCH_SIZE=512
  AUTO_PREFETCH_FACTOR=2
  MAX_WORKERS=6
elif [[ "${NGPUS}" -gt 0 ]]; then
  AUTO_BATCH_SIZE=256
  AUTO_PREFETCH_FACTOR=2
  MAX_WORKERS=4
else
  AUTO_BATCH_SIZE=256
  AUTO_PREFETCH_FACTOR=2
  MAX_WORKERS=4
fi

if [[ "${WORKERS_PER_RANK}" -gt "${MAX_WORKERS}" ]]; then
  WORKERS_PER_RANK="${MAX_WORKERS}"
fi

TARGET_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${AUTO_BATCH_SIZE}}"
if [[ -n "${BATCH_SIZE:-}" ]]; then
  # Explicit BATCH_SIZE remains per-rank for manual stress/throughput tuning.
  BATCH_SIZE="${BATCH_SIZE}"
elif [[ "${NGPUS}" -gt 1 ]]; then
  # Keep optimizer-step count close to the single-card recipe by preserving
  # the default global batch size and splitting it across DDP ranks.
  BATCH_SIZE=$(( (TARGET_GLOBAL_BATCH_SIZE + NGPUS - 1) / NGPUS ))
  if [[ "${BATCH_SIZE}" -lt 1 ]]; then
    BATCH_SIZE=1
  fi
else
  BATCH_SIZE="${AUTO_BATCH_SIZE}"
fi
EFFECTIVE_NGPUS="${NGPUS}"
if [[ "${EFFECTIVE_NGPUS}" -lt 1 ]]; then
  EFFECTIVE_NGPUS=1
fi
EFFECTIVE_GLOBAL_BATCH_SIZE=$(( BATCH_SIZE * EFFECTIVE_NGPUS ))
NUM_WORKERS="${NUM_WORKERS:-${WORKERS_PER_RANK}}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-${AUTO_PREFETCH_FACTOR}}"
COMPILE_MODE="${COMPILE_MODE:-default}"
USE_COMPILE="${USE_COMPILE:-0}"
FULL_TRAIN="${FULL_TRAIN:-1}"
case "${FULL_TRAIN}" in
  1|true|TRUE|yes|YES|on|ON)
    FULL_TRAIN_ENABLED=1
    ;;
  *)
    FULL_TRAIN_ENABLED=0
    ;;
esac
if [[ -z "${NUM_EPOCHS+x}" ]]; then
  if [[ "${FULL_TRAIN_ENABLED}" == "1" ]]; then
    NUM_EPOCHS=4
  else
    NUM_EPOCHS=8
  fi
fi
if [[ -z "${PATIENCE+x}" ]]; then
  if [[ "${FULL_TRAIN_ENABLED}" == "1" ]]; then
    PATIENCE=0
  else
    PATIENCE=3
  fi
fi

echo "Resource auto config: gpu_mem_mb=${GPU_MEM_MB}, cpu_cores=${CPU_CORES}, target_global_batch_size=${TARGET_GLOBAL_BATCH_SIZE}, effective_global_batch_size=${EFFECTIVE_GLOBAL_BATCH_SIZE}, per_rank_batch_size=${BATCH_SIZE}, num_workers=${NUM_WORKERS}, prefetch_factor=${PREFETCH_FACTOR}, use_compile=${USE_COMPILE}, compile_mode=${COMPILE_MODE}, pin_memory=${PIN_MEMORY}"
echo "EXPERIMENT_PROOF V19_AsymSeqAHeavy_LongRecipe_NSDrop010: base=V18_ReservedCatIds_LongRecipe_NSDrop010 mainline + V16_AsymSeqAHeavy geometry, reserved_cat_ids=1, full_train=${FULL_TRAIN}, final_epoch=${NUM_EPOCHS}, global_batch_preserved=1, seq_query_counts=${SEQ_QUERY_COUNTS:-seq_a:4,seq_b:1,seq_c:1,seq_d:1}, d_model=114, num_heads=6"

# ── Training args ──
TRAIN_ARGS=(
    --experiment_name V19_AsymSeqAHeavy_LongRecipe_NSDrop010
    --ns_tokenizer_type rankmixer
    --user_ns_tokens 5
    --user_dense_tokens 4
    --item_ns_tokens 2
    --item_dense_tokens 1
    --num_queries 1
    --seq_query_counts "${SEQ_QUERY_COUNTS:-seq_a:4,seq_b:1,seq_c:1,seq_d:1}"
    --ns_groups_json ""
    --emb_skip_threshold 1000000
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --seq_encoder_type swiglu
    --seq_encoder_type_by_domain "${SEQ_ENCODER_TYPE_BY_DOMAIN:-seq_a:transformer}"
    --seq_max_lens "${SEQ_MAX_LENS:-seq_a:256,seq_b:256,seq_c:512,seq_d:256}"
    --use_abs_time_user_int
    --use_sequence_summary_dense
    --use_seq_trunc_dense
    --use_missing_indicator_dense
    --use_target_match_dense
    --use_reserved_categorical_ids
    --target_match_seq_fids "seq_c:47"
    --num_epochs "${NUM_EPOCHS}"
    --patience "${PATIENCE}"
    --use_amp
    --amp_dtype bf16

    --d_model 114
    --emb_dim 64
    --num_hyformer_blocks 2
    --num_heads 6
    --hidden_mult 8
    --batch_size "${BATCH_SIZE}"
    --seq_top_k 50

    # ── V14 changes vs V9: SE-Net + training-trick bundle ──
    # NO geometry change: token count, d_model, num_queries all identical to V9.
    # The bundle is intentionally treated as ONE experiment (not 5 separate
    # A/Bs) because each trick individually moves CV by ~0.0001-0.0003 which
    # is below noise floor at 5 epochs; together they should compound visibly.
    # If V14 wins, follow-up ablations (SE only, EMA only, cosine only) become
    # justified.
    --use_se_net
    --se_reduction 2
    --weight_decay "${WEIGHT_DECAY:-0.03}"
    --lr_schedule cosine
    --warmup_steps "${WARMUP_STEPS:-1000}"
    --use_ema
    --ema_decay "${EMA_DECAY:-0.9995}"
    --label_smoothing "${LABEL_SMOOTHING:-0.005}"

    # ── V14_PerTokenFFN: the ONLY change vs V14 mainline ──
    # Query Boosting now uses RankMixer's parameter-isolated per-token FFN
    # (each of the T tokens owns its own FFN weights) instead of one shared
    # FFN. Everything else (SE-Net + training-trick bundle, geometry, d_model,
    # num_queries, encoder) is byte-for-byte identical to V14 → clean A/B.
    --per_token_ffn

    # ── V15_SeqARecentTransformer_NSDrop010 ──
    # Submission-style candidate: isolated seq_a self-attention encoder plus
    # the LB-validated NS spatial dropout dose from V14_NSSpatialDrop010.
    # Drops whole user/item NS tokens during training only; dense tokens,
    # sequence tokens and predict() remain unchanged.
    --ns_spatial_dropout_p "${NS_SPATIAL_DROPOUT_P:-0.10}"

    # ── V15_SeqARecentTransformer: isolated seq_a attention encoder ──
    # Safe knob: keep the recent-window max_len unchanged, keep seq_b/c/d on
    # swiglu, and give only seq_a a standard self-attention encoder. This
    # targets the large seq_a 101-300 long-history bucket without mixing in
    # the dangerous max_len=512 old-token knob.
    #
    # ── V14_DenseRobust: robustify dense fids 62-66 ──
    # nan/inf -> 0, nonnegative log1p, clip, then optional scale by clip.
    --dense_robust_user_fids "${DENSE_ROBUST_USER_FIDS:-62,63,64,65,66}"
    --dense_robust_clip "${DENSE_ROBUST_CLIP:-16.0}"
)

if [[ "${USE_COMPILE}" == "1" ]]; then
  TRAIN_ARGS+=(--use_compile --compile_mode "${COMPILE_MODE}")
fi

if [[ "${FULL_TRAIN_ENABLED}" == "1" ]]; then
  TRAIN_ARGS+=(--full_train)
fi

TRAIN_LOG_FILE="${TRAIN_LOG_PATH:-${SCRIPT_DIR}}/v19_asym_seqaheavy_longrecipe_nsdrop010_train_stdout.log"
mkdir -p "$(dirname "${TRAIN_LOG_FILE}")" 2>/dev/null || true

# ── Launch ──
if [[ "${NGPUS}" -gt 1 ]]; then
  echo "Detected ${NGPUS} GPUs, launching DDP with torchrun"
  torchrun --standalone --nproc_per_node="${NGPUS}" \
    "${SCRIPT_DIR}/train.py" "${TRAIN_ARGS[@]}" "$@" 2>&1 | tee "${TRAIN_LOG_FILE}"
  RC=${PIPESTATUS[0]}
else
  echo "Single GPU / CPU mode"
  python3 -u "${SCRIPT_DIR}/train.py" "${TRAIN_ARGS[@]}" "$@" 2>&1 | tee "${TRAIN_LOG_FILE}"
  RC=${PIPESTATUS[0]}
fi

echo ""
echo "================================================================================"
echo "===  V19_AsymSeqAHeavy_LongRecipe_NSDrop010 TAIL ECHO  === re-printing FINAL_TRAINING_SUMMARY ="
echo "================================================================================"
awk '/=== .* FINAL TRAINING SUMMARY START ===/,/=== .* FINAL TRAINING SUMMARY END ===/' \
  "${TRAIN_LOG_FILE}" 2>/dev/null \
  || echo "(no FINAL_TRAINING_SUMMARY block found -- training likely crashed before summary)"
# Also surface the DenseRobust lines + final val_auc trajectory near the
# very tail so the key A/B signal lands inside the last ~1000 log lines.
echo "--------------------------------------------------------------------------------"
echo "===  V19_AsymSeqAHeavy_LongRecipe_NSDrop010 KEY MONITORS (full-train final checkpoint + ReservedCatIds mainline + V16 asym seq_a heavy geometry + NSDrop) ==="
echo "--------------------------------------------------------------------------------"
grep -E "V19_AsymSeqAHeavy|EXPERIMENT_PROOF|full_train|FullTrain|Full-train|full-train|final checkpoint|target_global_batch_size|effective_global_batch_size|per_rank_batch_size|V18_ReservedCatIds|ReservedCatIds|use_reserved_categorical_ids|reserved categorical|raw_id\\+3|seq_query_counts|query map|total_query_tokens|seq_max_lens|num_epochs|patience|weight_decay|warmup_steps|ema_decay|label_smoothing|V15_SeqARecentTransformer|NSSpatialDrop|ns_spatial_dropout_p|NS spatial|seq_encoder_type_by_domain|seq encoder map|DenseRobust|PerTokenFFN ENABLED|Dense params:|Sparse params:|Total parameters" "${TRAIN_LOG_FILE}" 2>/dev/null | tail -n 220
echo "...."
grep -E "Epoch [0-9]+ (Validation \| AUC|FullTrain|ERank)|full_train|full-train|final checkpoint|ns_spatial_dropout_p" "${TRAIN_LOG_FILE}" 2>/dev/null | tail -n 100
echo "================================================================================"
echo "===  V19_AsymSeqAHeavy_LongRecipe_NSDrop010 job exit code = ${RC}      ==="
echo "================================================================================"
exit ${RC}

# ============================================================================
# V14_PerTokenFFN = V14 + parameter-isolated per-token Query-Boosting FFN
# ----------------------------------------------------------------------------
# WHY: HyFormer's Query Boosting is RankMixer (token mixing + Per-Token FFN).
# RankMixer's Per-Token FFN gives EACH of the T tokens its OWN FFN weights
# ("isolating the parameters for each token, rather than sharing parameters") —
# that parameter isolation is the capacity-scaling mechanism. V14's
# RankMixerBlock used ONE shared FFN for all tokens, which silently removed
# that mechanism and is the leading suspect for the stalled scaling
# (CV up / LB down, capacity not translating to gains).
#
# WHAT CHANGED (clean A/B — only this):
#   * model.py RankMixerBlock: per_token_ffn=True path → per-token W1/b1/W2/b2
#     parameters, applied via einsum. Token mixing, LayerNorms, GELU, dropout,
#     residual, post_norm all byte-for-byte identical to V14. Init magnitude
#     matched to nn.Linear default so only isolation differs.
#   * train.py: new --per_token_ffn flag (default OFF = exact V14 baseline).
#   * run.sh:   --per_token_ffn turned ON. Nothing else touched.
#
# CAPACITY DELTA (d_model=80, T=20, hidden_mult=8 → H=640, 2 blocks):
#   Query-Boosting FFN params/block: ~103K (shared) → ~2.06M (per-token).
#   Watch "Dense params:" / "Total parameters" in the log to confirm.
#
# MONITORING (key A/B signals, all tail-echoed within the last ~1000 lines):
#   * "PerTokenFFN ENABLED ..." line (capacity delta, printed at model build)
#   * Dense/Sparse/Total param counts (confirms the capacity jump landed)
#   * Per-epoch "Epoch N Validation | AUC: ..." trajectory
#   * Per-epoch "Epoch N ERank | ..." — erank/{ns,block_out_i,query_concat,output}.
#     HYPOTHESIS: if the shared FFN was the bottleneck, block_out/output erank
#     should rise here and val_auc should beat V14's 0.8384 CV baseline.
#   * FINAL_TRAINING_SUMMARY block (per-epoch table + best AUC + baseline overlay)
#
# COMPARE AGAINST V14 mainline CV val_auc=0.8384. To overlay automatically:
#   V14_BASELINE_AUC="<v14 per-epoch aucs csv>" bash run.sh
# To reproduce the exact V14 baseline from THIS code, drop --per_token_ffn.
# ============================================================================
#
# ---- inherited V14 notes: V9 + SE-Net + training-trick bundle ----
# Single bundle vs V9: SE-Net per-NS-token gating + AdamW weight_decay 0.02 +
# cosine LR with 500 warmup steps + ModelEMA on dense params (decay 0.999) +
# label_smoothing 0.01 on BCE targets.
#
# Why a bundle instead of 5 A/Bs: each trick individually is in the
# +0.0001-0.0003 range, below 5-epoch noise. The bundle treats them as
# combined regularization+optimization improvements, which is also how
# every top github submission ships them. If V14 wins clean, we then ablate
# (SE-only / EMA-only / cosine-only) to attribute.
#
# Module references:
#   * SENetGating          model.py (ported from github/7/best:1336-1364)
#   * ModelEMA             utils.py (ported from github/7/best:236-286)
#   * Cosine LR + warmup   trainer.py LambdaLR (warmup -> half-cosine to 0.05*lr)
#   * Label smoothing      trainer.py _train_step (y = (1-ls)*y + 0.5*ls)
#   * weight_decay         passed to AdamW(dense) only; sparse Adagrad unchanged
#
# EMA semantics: shadow updated AFTER each successful optimizer step. At
# evaluate() time the shadow weights are swapped in (apply_shadow), eval runs,
# then training weights restored. Sparse Embedding params are EXCLUDED from
# EMA (they have their own Adagrad cold-restart dynamics that would conflict).
#
# Monitoring (inherited from V11): per-epoch val_auc/val_logloss/loss/throughput,
# erank/{ns,qc,output,b0,b1}, target_match per-domain hit_rate, seq-usage
# per-domain. FINAL_TRAINING_SUMMARY + tail-echo.
#
# Optional baseline overlay:
#   V14_BASELINE_AUC="0.832,0.835,0.836,0.836,0.835" bash run.sh
