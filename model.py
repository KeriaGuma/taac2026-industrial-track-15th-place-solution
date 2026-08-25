"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_gap_buckets: Optional[dict] = None  # {domain: tensor [B, L]}
    seq_session_buckets: Optional[dict] = None  # {domain: tensor [B, L]}


_MONO_TIME_BUCKET_SECONDS = (
    0,
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            attn_bias: Optional additive bias broadcastable to
                (B, num_heads, Lq, Lk). Negative values suppress keys.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        if sdpa_attn_mask is not None:
            # Some rows can have an all-padding sequence. SDPA returns NaN for an
            # all-masked query row, so expose the first key as a harmless zeroed
            # value instead; downstream residuals keep the representation sane.
            no_valid_key = ~sdpa_attn_mask.any(dim=-1, keepdim=True)
            if no_valid_key.any():
                sdpa_attn_mask = sdpa_attn_mask.clone()
                sdpa_attn_mask[..., :1] = sdpa_attn_mask[..., :1] | no_valid_key

        final_attn_mask = sdpa_attn_mask
        if attn_bias is not None:
            bias = attn_bias.to(device=Q.device, dtype=Q.dtype)
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)
            if bias.shape[1] == 1 and self.num_heads != 1:
                bias = bias.expand(-1, self.num_heads, -1, -1)
            if final_attn_mask is not None:
                mask_bias = torch.zeros(
                    final_attn_mask.shape, device=Q.device, dtype=Q.dtype)
                mask_bias = mask_bias.masked_fill(~final_attn_mask, -1.0e4)
                final_attn_mask = mask_bias + bias
            else:
                final_attn_mask = bias

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=final_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.
            attn_bias: Optional additive attention bias for keys.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attn_bias=attn_bias,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: feedforward network. When per_token_ffn=True each token
       position owns an independent FFN (RankMixer's parameter-isolated design);
       otherwise a single FFN is shared across all tokens (V14 baseline).
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full',  # 'full' | 'ffn_only' | 'none'
        per_token_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode
        self.per_token_ffn = bool(per_token_ffn)

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        hidden_dim = d_model * hidden_mult
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

        if self.per_token_ffn:
            # ── HyFormer/RankMixer FIX: parameter-isolated Per-Token FFN ──
            # Each of the T token positions owns an INDEPENDENT 2-layer FFN
            # (D -> H -> D). This is RankMixer's capacity-scaling mechanism
            # ("per-token FFN ... isolating the parameters for each token, rather
            # than sharing parameters across all tokens"). The V14 baseline used
            # a single SHARED FFN -- that is exactly the deviation this version
            # repairs. Token mixing, norms, dropout and the residual are kept
            # byte-for-byte identical so this is a clean A/B on parameter
            # isolation alone.
            self.fc1 = None
            self.fc2 = None
            self.W1 = nn.Parameter(torch.empty(n_total, d_model, hidden_dim))
            self.b1 = nn.Parameter(torch.zeros(n_total, hidden_dim))
            self.W2 = nn.Parameter(torch.empty(n_total, hidden_dim, d_model))
            self.b2 = nn.Parameter(torch.zeros(n_total, d_model))
            # Match nn.Linear default init scale (kaiming_uniform a=sqrt(5)) so the
            # ONLY thing that changes vs baseline is parameter isolation, not the
            # init magnitude.
            bound1 = 1.0 / math.sqrt(d_model)
            bound2 = 1.0 / math.sqrt(hidden_dim)
            nn.init.uniform_(self.W1, -bound1, bound1)
            nn.init.uniform_(self.b1, -bound1, bound1)
            nn.init.uniform_(self.W2, -bound2, bound2)
            nn.init.uniform_(self.b2, -bound2, bound2)
        else:
            # Baseline (V14): single FFN shared across all tokens.
            self.fc1 = nn.Linear(d_model, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN (parameter-isolated) or shared FFN (baseline)
        x = self.norm(Q_hat)
        if self.per_token_ffn:
            # x: (B, T, D); W1: (T, D, H) -> h: (B, T, H). Each token t is
            # multiplied by its OWN weight slice W1[t]; bias broadcasts over batch.
            h = torch.einsum('btd,tdh->bth', x, self.W1) + self.b1
            h = F.gelu(h)
            h = self.dropout(h)
            # h: (B, T, H); W2: (T, H, D) -> Q_e: (B, T, D)
            Q_e = torch.einsum('bth,thd->btd', h, self.W2) + self.b2
        else:
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        seq_query_counts: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        if seq_query_counts is None:
            seq_query_counts = [num_queries] * num_sequences
        if len(seq_query_counts) != num_sequences:
            raise ValueError(
                f"seq_query_counts length {len(seq_query_counts)} != num_sequences {num_sequences}"
            )
        if any(int(n) <= 0 for n in seq_query_counts):
            raise ValueError(f"seq_query_counts must be positive, got {seq_query_counts}")
        self.seq_query_counts = [int(n) for n in seq_query_counts]

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has its own number of independent FFN-generated query tokens.
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(self.seq_query_counts[i])
            ])
            for i in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


class RecentCoreTailEncoder(nn.Module):
    """Short-window heavy encoder with a cheap long-tail summary token.

    For long sequences it keeps only the latest ``core_k`` valid tokens on the
    expensive self-attention path and prepends one mean-pooled tail token from
    older valid events. Once the sequence is already compressed to
    ``core_k + tail`` length, later HyFormer blocks simply refine that compact
    representation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        core_k: int = 64,
        depth: int = 3,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        use_tail: bool = True,
    ) -> None:
        super().__init__()
        self.core_k = max(1, int(core_k))
        self.depth = max(1, int(depth))
        self.use_tail = bool(use_tail)
        self.layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
            for _ in range(self.depth)
        ])
        self.tail_norm = nn.LayerNorm(d_model)

    def _gather_recent_core(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        device = x.device
        valid_len = (~key_padding_mask).sum(dim=1)
        actual_k = torch.clamp(valid_len, max=self.core_k)
        start_pos = valid_len - actual_k

        offsets = torch.arange(self.core_k, device=device).unsqueeze(0).expand(B, -1)
        indices = torch.clamp(start_pos.unsqueeze(1) + offsets, min=0, max=L - 1)
        gathered = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))

        pad_count = self.core_k - actual_k
        pos = torch.arange(self.core_k, device=device).unsqueeze(0)
        core_mask = pos < pad_count.unsqueeze(1)
        gathered = gathered * (~core_mask).unsqueeze(-1).to(gathered.dtype)
        return gathered, core_mask

    def _tail_token(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        device = x.device
        valid = ~key_padding_mask
        valid_len = valid.sum(dim=1)
        start_pos = torch.clamp(valid_len - self.core_k, min=0)
        pos = torch.arange(L, device=device).unsqueeze(0)
        tail_valid = valid & (pos < start_pos.unsqueeze(1))
        denom = tail_valid.sum(dim=1).clamp_min(1).to(x.dtype)
        tail = (x * tail_valid.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom.unsqueeze(-1)
        tail = self.tail_norm(tail).unsqueeze(1)
        tail_mask = ~tail_valid.any(dim=1, keepdim=True)
        tail = tail * (~tail_mask).unsqueeze(-1).to(x.dtype)
        return tail, tail_mask

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(
                x.shape[:2], device=x.device, dtype=torch.bool)

        compressed_len = self.core_k + (1 if self.use_tail else 0)
        if x.shape[1] > compressed_len:
            core, core_mask = self._gather_recent_core(x, key_padding_mask)
            if self.use_tail:
                tail, tail_mask = self._tail_token(x, key_padding_mask)
                x = torch.cat([tail, core], dim=1)
                key_padding_mask = torch.cat([tail_mask, core_mask], dim=1)
            else:
                x = core
                key_padding_mask = core_mask

        for layer in self.layers:
            x, key_padding_mask = layer(
                x, key_padding_mask=key_padding_mask,
                rope_cos=None, rope_sin=None)
        return x, key_padding_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False,
    recent_core_k: int = 64,
    recent_core_depth: int = 3,
    recent_core_tail: bool = True,
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', 'longer', or
            'recent_core_tail'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    elif encoder_type == 'recent_core_tail':
        return RecentCoreTailEncoder(
            d_model=d_model,
            num_heads=num_heads,
            core_k=recent_core_k,
            depth=recent_core_depth,
            hidden_mult=hidden_mult,
            dropout=dropout,
            use_tail=recent_core_tail,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        per_token_ffn: bool = False,
        seq_encoder_types: Optional[List[str]] = None,
        recent_core_k: int = 64,
        recent_core_depth: int = 3,
        recent_core_tail: bool = True,
        seq_query_counts: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns
        if seq_query_counts is None:
            seq_query_counts = [num_queries] * num_sequences
        if len(seq_query_counts) != num_sequences:
            raise ValueError(
                f"seq_query_counts length {len(seq_query_counts)} != num_sequences {num_sequences}"
            )
        if any(int(n) <= 0 for n in seq_query_counts):
            raise ValueError(f"seq_query_counts must be positive, got {seq_query_counts}")
        self.seq_query_counts = [int(n) for n in seq_query_counts]
        if seq_encoder_types is None:
            seq_encoder_types = [seq_encoder_type] * num_sequences
        if len(seq_encoder_types) != num_sequences:
            raise ValueError(
                f"seq_encoder_types length {len(seq_encoder_types)} != num_sequences {num_sequences}"
            )

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_types[i],
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal,
                recent_core_k=recent_core_k,
                recent_core_depth=recent_core_depth,
                recent_core_tail=recent_core_tail,
            )
            for i in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = sum(per-domain query counts) + Nns.
        n_total = sum(self.seq_query_counts) + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode,
            per_token_ffn=per_token_ffn,
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
        seq_time_biases: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.
            seq_time_biases: Optional list of additive key biases, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            attn_bias = None
            if seq_time_biases is not None:
                attn_bias = seq_time_biases[i]
                if attn_bias is not None and attn_bias.shape[-1] != next_seqs[i].shape[1]:
                    # Length-changing encoders such as RecentCoreTail compress
                    # the sequence. Skip stale full-length time biases rather
                    # than applying them to the wrong keys.
                    attn_bias = None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs, attn_bias=attn_bias,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, sum_q + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, sum_q + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            n_q = self.seq_query_counts[i]
            next_q_list.append(boosted[:, offset:offset + n_q, :])
            offset += n_q
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class DenseChunkTokenizer(nn.Module):
    """Split dense features into several projected NS tokens."""

    def __init__(
        self,
        dense_dim: int,
        d_model: int,
        num_tokens: int,
    ) -> None:
        super().__init__()
        if dense_dim <= 0:
            raise ValueError("dense_dim must be positive")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        self.dense_dim = int(dense_dim)
        self.num_tokens = int(num_tokens)
        self.chunk_dim = math.ceil(self.dense_dim / self.num_tokens)
        self.padded_dim = self.chunk_dim * self.num_tokens
        self._pad_size = self.padded_dim - self.dense_dim
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(self.num_tokens)
        ])
        logging.info(
            f"DenseChunkTokenizer: dense_dim={dense_dim}, "
            f"num_tokens={num_tokens}, chunk_dim={self.chunk_dim}, "
            f"pad={self._pad_size}")

    def forward(self, dense_feats: torch.Tensor) -> torch.Tensor:
        if self._pad_size > 0:
            dense_feats = F.pad(dense_feats, (0, self._pad_size))
        chunks = dense_feats.split(self.chunk_dim, dim=-1)
        tokens = [
            F.silu(proj(chunk)).unsqueeze(1)
            for chunk, proj in zip(chunks, self.token_projs)
        ]
        return torch.cat(tokens, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# V14: Squeeze-and-Excitation gating for NS tokens.
# Ported from github/7/best/model.py:1336-1364 (SENetGating).
# ═══════════════════════════════════════════════════════════════════════════════


class SENetGating(nn.Module):
    """Squeeze-and-Excitation gating for NS tokens.

    Per-token importance via a two-layer bottleneck MLP on the mean-pooled
    (squeezed) token representation. Returns x * sigmoid(MLP(mean(x))).
    Token COUNT is preserved.
    """

    def __init__(self, num_tokens: int, d_model: int, reduction: int = 2) -> None:
        super().__init__()
        mid = max(num_tokens // reduction, 4)
        self.gate = nn.Sequential(
            nn.Linear(num_tokens, mid),
            nn.ReLU(),
            nn.Linear(mid, num_tokens),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE gating.

        Args:
            x: (B, T, D) NS tokens.

        Returns:
            (B, T, D), gated.
        """
        w = x.mean(dim=-1)             # (B, T)  -- squeeze
        w = self.gate(w)                # (B, T)  -- excitation
        return x * w.unsqueeze(-1)      # (B, T, D) -- scale


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        user_dense_tokens: int = 1,
        item_dense_tokens: int = 1,
        # ── V14: SE-Net per-token gating ──
        use_se_net: bool = False,
        se_reduction: int = 2,
        # ── V14_PerTokenFFN: parameter-isolated Query-Boosting FFN ──
        per_token_ffn: bool = False,
        # ── V14_NSSpatialDrop: token-wise NS embedding dropout ──
        ns_spatial_dropout_p: float = 0.0,
        # ── V15_SeqARecentTransformer: per-domain encoder override ──
        seq_encoder_type_by_domain: str = '',
        # ── V16_InterEventGapSession: per-token gap/session embeddings ──
        use_inter_event_gap_session: bool = False,
        # ── V16_MonoTimeDecayAttnBias: forced monotone recent-first cross-attn ──
        use_mono_time_decay_attn_bias: bool = False,
        mono_time_decay_domains: str = 'seq_a,seq_b',
        mono_time_decay_lambda_min: float = 0.02,
        mono_time_decay_lambda_init: float = 0.02,
        # ── V17_RecentCoreTail: short-window heavy seq encoder ──
        recent_core_k: int = 64,
        recent_core_depth: int = 3,
        recent_core_tail: bool = True,
        # ── V16_AsymSeqAHeavy: per-domain query-token allocation ──
        seq_query_counts: str = '',
    ) -> None:
        super().__init__()
        self.per_token_ffn = bool(per_token_ffn)
        self.ns_spatial_dropout_p = float(ns_spatial_dropout_p)
        self.use_inter_event_gap_session = bool(use_inter_event_gap_session)
        self.use_mono_time_decay_attn_bias = bool(use_mono_time_decay_attn_bias)
        self.recent_core_k = max(1, int(recent_core_k))
        self.recent_core_depth = max(1, int(recent_core_depth))
        self.recent_core_tail = bool(recent_core_tail)
        if self.ns_spatial_dropout_p < 0.0 or self.ns_spatial_dropout_p >= 1.0:
            raise ValueError(
                f"ns_spatial_dropout_p must be in [0, 1), got {self.ns_spatial_dropout_p}"
            )

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        domain_to_idx = {d: i for i, d in enumerate(self.seq_domains)}
        self.mono_time_decay_domains = {
            d.strip() for d in str(mono_time_decay_domains or '').replace(';', ',').split(',')
            if d.strip()
        }
        unknown_decay_domains = self.mono_time_decay_domains - set(self.seq_domains)
        if unknown_decay_domains:
            raise ValueError(
                f"Unknown mono_time_decay_domains {sorted(unknown_decay_domains)}; "
                f"expected subset of {self.seq_domains}"
            )
        self.mono_time_decay_lambda_min = max(0.0, float(mono_time_decay_lambda_min))
        self.seq_encoder_type_by_domain = str(seq_encoder_type_by_domain or '')
        self.seq_encoder_types = [seq_encoder_type] * self.num_sequences
        if self.seq_encoder_type_by_domain:
            for raw in self.seq_encoder_type_by_domain.replace(';', ',').split(','):
                raw = raw.strip()
                if not raw:
                    continue
                if ':' not in raw:
                    raise ValueError(
                        "seq_encoder_type_by_domain entries must look like "
                        f"'seq_a:transformer', got {raw!r}"
                    )
                domain, encoder = [part.strip() for part in raw.split(':', 1)]
                if domain not in domain_to_idx:
                    raise ValueError(
                        f"Unknown sequence domain {domain!r}; expected one of {self.seq_domains}"
                    )
                if encoder not in ('swiglu', 'transformer', 'longer', 'recent_core_tail'):
                    raise ValueError(
                        f"Unknown encoder type {encoder!r} for {domain}; "
                        "expected swiglu/transformer/longer/recent_core_tail"
                    )
                self.seq_encoder_types[domain_to_idx[domain]] = encoder
        logging.info(
            "V15_SeqARecentTransformer seq encoder map: %s",
            dict(zip(self.seq_domains, self.seq_encoder_types)),
        )
        self.seq_query_counts_spec = str(seq_query_counts or '')
        self.seq_query_counts = [int(num_queries)] * self.num_sequences
        if self.seq_query_counts_spec:
            for raw in self.seq_query_counts_spec.replace(';', ',').split(','):
                raw = raw.strip()
                if not raw:
                    continue
                if ':' not in raw:
                    raise ValueError(
                        "seq_query_counts entries must look like 'seq_a:4', "
                        f"got {raw!r}"
                    )
                domain, count = [part.strip() for part in raw.split(':', 1)]
                if domain not in domain_to_idx:
                    raise ValueError(
                        f"Unknown sequence domain {domain!r}; expected one of {self.seq_domains}"
                    )
                count_i = int(count)
                if count_i <= 0:
                    raise ValueError(
                        f"seq_query_counts for {domain} must be positive, got {count_i}"
                    )
                self.seq_query_counts[domain_to_idx[domain]] = count_i
        self.total_query_tokens = int(sum(self.seq_query_counts))
        logging.info(
            "V19_AsymSeqAHeavy query map: %s total_query_tokens=%d",
            dict(zip(self.seq_domains, self.seq_query_counts)),
            self.total_query_tokens,
        )
        self.num_time_buckets = num_time_buckets
        if self.use_inter_event_gap_session and self.num_time_buckets <= 0:
            raise ValueError("use_inter_event_gap_session requires num_time_buckets > 0")
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.user_dense_tokens = max(1, int(user_dense_tokens))
        self.item_dense_tokens = max(1, int(item_dense_tokens))
        if self.ns_spatial_dropout_p > 0:
            logging.info(
                "V15_SeqARecentTransformer + V14_NSSpatialDrop ENABLED: p=%.4f, "
                "applies to user/item NS tokens only during training; "
                "dense/sequence tokens and predict() unchanged",
                self.ns_spatial_dropout_p,
            )
        age_hours = torch.tensor(_MONO_TIME_BUCKET_SECONDS, dtype=torch.float32) / 3600.0
        self.register_buffer(
            'mono_time_decay_log_age',
            torch.log1p(age_hours),
            persistent=False,
        )
        init_extra = max(1e-6, float(mono_time_decay_lambda_init) - self.mono_time_decay_lambda_min)
        init_raw = math.log(math.expm1(init_extra))
        self.mono_time_decay_raw = nn.Parameter(
            torch.full((self.num_sequences,), init_raw, dtype=torch.float32)
        )
        if self.use_mono_time_decay_attn_bias:
            logging.info(
                "V16_MonoTimeDecayAttnBias ENABLED: domains=%s lambda_min=%.6f "
                "lambda_init=%.6f form=-lambda*log1p(age_hours), applied to cross-attn keys",
                sorted(self.mono_time_decay_domains),
                self.mono_time_decay_lambda_min,
                self.mono_time_decay_lambda_min + F.softplus(self.mono_time_decay_raw[0]).item(),
            )
        else:
            # The parameter is kept in the state_dict for compatibility, but
            # disabled runs must not expose it to DDP as a trainable unused param.
            self.mono_time_decay_raw.requires_grad_(False)
            logging.info("V16_MonoTimeDecayAttnBias disabled: frozen mono_time_decay_raw")
        if 'recent_core_tail' in self.seq_encoder_types:
            logging.info(
                "V17_RecentCoreTail ENABLED: k=%d depth=%d tail=%s encoder_map=%s",
                self.recent_core_k,
                self.recent_core_depth,
                self.recent_core_tail,
                dict(zip(self.seq_domains, self.seq_encoder_types)),
            )

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_tokenizer = DenseChunkTokenizer(
                dense_dim=user_dense_dim,
                d_model=d_model,
                num_tokens=self.user_dense_tokens,
            )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_tokenizer = DenseChunkTokenizer(
                dense_dim=item_dense_dim,
                d_model=d_model,
                num_tokens=self.item_dense_tokens,
            )

        # Total NS token count
        self.num_ns = (
            num_user_ns
            + (self.user_dense_tokens if self.has_user_dense else 0)
            + num_item_ns
            + (self.item_dense_tokens if self.has_item_dense else 0)
        )

        # ================== V14: SE-Net per-NS-token gating (optional) ==================
        # Applied to ns_tokens BEFORE the block stack. Token COUNT is
        # unchanged -- only re-weighted per (B, t). Cheap.
        self.use_se_net = bool(use_se_net)
        if self.use_se_net:
            self.se_gate = SENetGating(
                num_tokens=self.num_ns,
                d_model=d_model,
                reduction=se_reduction,
            )
            logging.info(
                f"V14 SENetGating enabled: num_tokens={self.num_ns}, "
                f"d_model={d_model}, reduction={se_reduction}"
            )
        else:
            self.se_gate = None

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = self.total_query_tokens + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=sum(seq_query_counts)+num_ns="
                f"{self.total_query_tokens}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)
        if self.use_inter_event_gap_session:
            self.gap_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)
            self.session_embedding = nn.Embedding(3, d_model, padding_idx=0)
            logging.info(
                "V16_InterEventGapSession ENABLED: per-token gap buckets "
                "+ session boundary embedding, num_gap_buckets=%d",
                num_time_buckets,
            )

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
            seq_query_counts=self.seq_query_counts,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                seq_encoder_types=self.seq_encoder_types,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                per_token_ffn=per_token_ffn,
                recent_core_k=self.recent_core_k,
                recent_core_depth=self.recent_core_depth,
                recent_core_tail=self.recent_core_tail,
                seq_query_counts=self.seq_query_counts,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ── V14_PerTokenFFN monitoring: make the capacity change explicit ──
        if per_token_ffn:
            hidden_dim = d_model * hidden_mult
            # Per block: T independent (D->H) + (H->D) FFNs (weights + biases).
            per_block = T * (d_model * hidden_dim + hidden_dim
                             + hidden_dim * d_model + d_model)
            shared_block = (d_model * hidden_dim + hidden_dim
                            + hidden_dim * d_model + d_model)
            logging.info(
                f"PerTokenFFN ENABLED: T={T} parameter-isolated FFNs per block "
                f"(D={d_model} -> H={hidden_dim} -> D); "
                f"QueryBoosting FFN params/block {shared_block:,} (shared) -> "
                f"{per_block:,} (per-token), x{num_hyformer_blocks} blocks."
            )

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection (Phase 1): keep dim at Nq*S*D so erank/output is
        # no longer hard-capped at d_model. Was Linear(640->80); now 640->640
        # with LN + Dropout so downstream scaling experiments can actually
        # raise effective rank past the old D=80 ceiling.
        proj_dim = self.total_query_tokens * d_model
        self.proj_dim = proj_dim
        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier (Phase 1): input dim widened from d_model to proj_dim
        self.clsfier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _apply_ns_spatial_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Drop whole NS tokens independently per sample during training."""
        p = self.ns_spatial_dropout_p
        if (not self.training) or p <= 0.0:
            return x
        keep = 1.0 - p
        mask = (
            torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < keep
        ).to(dtype=x.dtype)
        return x * mask / keep

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0
        if self.use_inter_event_gap_session:
            nn.init.xavier_normal_(self.gap_embedding.weight.data)
            self.gap_embedding.weight.data[0, :] = 0
            nn.init.xavier_normal_(self.session_embedding.weight.data)
            self.session_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1
        if self.use_inter_event_gap_session:
            skip_count += 2

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        gap_bucket_ids: Optional[torch.Tensor] = None,
        session_bucket_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
        if self.use_inter_event_gap_session:
            if gap_bucket_ids is not None:
                token_emb = token_emb + self.gap_embedding(gap_bucket_ids)
            if session_bucket_ids is not None:
                token_emb = token_emb + self.session_embedding(session_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _build_mono_time_decay_biases(
        self,
        seq_time_buckets: dict,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if not self.use_mono_time_decay_attn_bias:
            return None

        log_age_table = self.mono_time_decay_log_age
        lambdas = self.mono_time_decay_lambda_min + F.softplus(self.mono_time_decay_raw)
        biases: List[Optional[torch.Tensor]] = []
        for i, domain in enumerate(self.seq_domains):
            tb = seq_time_buckets[domain].clamp(
                min=0, max=log_age_table.numel() - 1)
            log_age = log_age_table.to(device=tb.device)[tb].to(torch.float32)
            enabled = 1.0 if domain in self.mono_time_decay_domains else 0.0
            bias = -(lambdas[i].to(log_age.device) * enabled) * log_age
            biases.append(bias.unsqueeze(1).unsqueeze(1))  # (B, 1, 1, L)
        return biases

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        seq_time_biases: Optional[List[Optional[torch.Tensor]]] = None,
        apply_dropout: bool = True,
        return_intermediates: bool = False,
    ):
        """Runs the multi-sequence block stack with dropout and output projection.

        When return_intermediates=True returns (output, intermediates) where
        intermediates is a dict containing per-block query concats and the
        pre-output_proj query_concat tensor — used by the Phase 0 erank monitor.
        """
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        block_outs = [] if return_intermediates else None

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
                seq_time_biases=seq_time_biases,
            )

            if return_intermediates:
                B = curr_qs[0].shape[0]
                block_outs.append(torch.cat(curr_qs, dim=1).reshape(B, -1))

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, sum_q, D)
        query_concat = all_q.view(B, -1)  # (B, sum_q*D) -- pre-output_proj signal
        output = self.output_proj(query_concat)  # (B, proj_dim) — was (B, D)

        if return_intermediates:
            intermediates = {
                'block_outs': block_outs,        # list of (B, sum_q*D)
                'query_concat': query_concat,    # (B, sum_q*D)
            }
            return output, intermediates

        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)   # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)   # (B, num_item_groups, D)
        user_ns = self._apply_ns_spatial_dropout(user_ns)
        item_ns = self._apply_ns_spatial_dropout(item_ns)

        ns_parts = [user_ns]
        if self.has_user_dense:
            ns_parts.append(self.user_dense_tokenizer(inputs.user_dense_feats))
        ns_parts.append(item_ns)
        if self.has_item_dense:
            ns_parts.append(self.item_dense_tokenizer(inputs.item_dense_feats))

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # V14: SE-Net per-token gating; preserves token count.
        if self.se_gate is not None:
            ns_tokens = self.se_gate(ns_tokens)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            gap_bucket = None
            session_bucket = None
            if inputs.seq_gap_buckets is not None:
                gap_bucket = inputs.seq_gap_buckets.get(domain)
            if inputs.seq_session_buckets is not None:
                session_bucket = inputs.seq_session_buckets.get(domain)
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                gap_bucket,
                session_bucket)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)
        seq_time_biases = self._build_mono_time_decay_biases(inputs.seq_time_buckets)

        # 4. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            seq_time_biases=seq_time_biases,
            apply_dropout=self.training
        )

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            ns_parts.append(self.user_dense_tokenizer(inputs.user_dense_feats))
        ns_parts.append(item_ns)
        if self.has_item_dense:
            ns_parts.append(self.item_dense_tokenizer(inputs.item_dense_feats))

        ns_tokens = torch.cat(ns_parts, dim=1)

        # V14: SE-Net per-token gating; preserves token count.
        if self.se_gate is not None:
            ns_tokens = self.se_gate(ns_tokens)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            gap_bucket = None
            session_bucket = None
            if inputs.seq_gap_buckets is not None:
                gap_bucket = inputs.seq_gap_buckets.get(domain)
            if inputs.seq_session_buckets is not None:
                session_bucket = inputs.seq_session_buckets.get(domain)
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                gap_bucket,
                session_bucket)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)
        seq_time_biases = self._build_mono_time_decay_biases(inputs.seq_time_buckets)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            seq_time_biases=seq_time_biases,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return logits, output
