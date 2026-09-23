# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B MLA + DSA (Deep Sparse Attention) indexer model for vLLM-Ascend.

This model is the Qwen3.5-9B MLA-adapted model (see ``qwen3_5_mla.py``) with an
additional DSA indexer module attached to every full-attention (MLA) layer.

The MLA half is a verbatim copy of ``qwen3_5_mla.py`` (same KV-compressed
architecture, Q-proj gate, RoPE-first, GQA kv_b_proj broadcast, NeoX RoPE).
On top of it, each full-attention layer carries a lightweight learned indexer
(structurally identical to DeepSeek-V3.2's ``Indexer``: ``wq_b`` / ``wk`` /
``weights_proj`` / ``k_norm`` + a K-only paged cache) that selects the top-k
keys each query should attend to.  The main MLA attention then attends only to
those selected keys via vLLM-Ascend's SFA path (``AscendSFAImpl`` +
``npu_lightning_indexer`` + ``npu_sparse_flash_attention`` ACLNN ops).

Key differences from the plain MLA model (``qwen3_5_mla.py``):

1. Every full-attention layer builds a ``Qwen3_5Indexer`` and passes
   ``is_sparse=True`` / ``indexer`` / ``topk_indices_buffer`` into
   ``MLAModules`` so the SFA backend (``AscendSFABackend`` / ``AscendSFAImpl``)
   is selected instead of the plain MLA backend.
2. The attention impl is replaced with ``Qwen3_5DSAImpl``
   (``Qwen3_5DSAImplMixin`` + ``Qwen3_5MLAImplMixin`` + ``AscendSFAImpl``):
   a self-contained ``forward`` that runs the Qwen MLA preprocess (kv_a_proj,
   q_proj+gate, NeoX RoPE, GQA weight absorption) and delegates the indexer
   top-k selection + sparse attention to the reusable ``AscendSFAImpl``
   helpers (``indexer_select_pre_process`` / ``indexer_select_post_process`` /
   ``_execute_sparse_flash_attention_process`` / ``_v_up_proj``).
3. The indexer's ``wq_b`` takes ``hidden_states`` directly (Qwen3.5 has no
   Q-lora residual, unlike DeepSeek-V3.2 whose indexer consumes the Q latent),
   so ``indexer_select_post_process`` is overridden to feed ``hidden`` (not
   ``q_c``) into ``wq_b``.
4. The checkpoint stores ``indexer.wk`` and ``indexer.weights_proj`` as two
   separate weights; ``load_weights`` fuses them into a single
   ``wk_weights_proj`` (matching the fused GEMM the SFA impl expects).

Differences from Qwen3.6-35B-A3B MLA:
- Dense MLP (Qwen3NextMLP) instead of MoE (Qwen3NextSparseMoeBlock)
- num_kv_heads = 4 (instead of 2)
- hidden_size = 4096 (instead of 2048)
- model_type = "qwen3_5_text" (instead of "qwen3_5_moe_text")
- No QwenNextMixtureOfExperts mixin, no set_moe_parameters
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.models.qwen3_5 import Qwen3_5RMSNorm
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)

# NOTE: vllm_ascend.attention.* imports are done lazily (inside methods) to
# avoid a circular-import chain: mla_v1 → attention_v1 → device_op →
# ops.fused_moe → experts_selector → device_op.
from vllm_ascend.memcache_comm_fence import record_attention_compute_start
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
from vllm_ascend.utils import maybe_trans_nz

from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DSAForCausalLM,
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
)

if TYPE_CHECKING:
    from vllm_ascend.attention.mla_v1 import AscendMLAImpl

logger = init_logger(__name__)

# ────────────────────────────────────────────────────────────────────
#  MLA dimension constants derived from the Qwen3.6 compressed config
# ────────────────────────────────────────────────────────────────────
# head_dim          = 256
# partial_rotary    = 0.25  →  qk_rope_head_dim = 64
# qk_nope_head_dim  = 256 - 64 = 192
# v_head_dim        = 256  (same as head_dim)
# kv_lora_rank      = 512  (kv_compression.rank)
# q_lora_rank       = None (no Q low-rank)
# num_heads         = 16,  num_kv_heads = 2


# ════════════════════════════════════════════════════════════════════
#  1.  Custom modules
# ════════════════════════════════════════════════════════════════════


class Qwen3_5KVProjWithMQA(nn.Module):
    """Wrap ``k_proj / k_norm / v_proj / kv_a_proj`` into a single module.

    Externally this module looks like DeepSeek's ``kv_a_proj_with_mqa``:
    ``hidden(2048) -> [latent(512), rope(64)]``.

    Internally it runs the original Qwen3.6 compressed-KV pipeline:

        k = k_proj(hidden)            # [N, 512]  = [2*256]
        k = k_norm(k.view(N,2,256))   # RMSNorm per-head
        k_rot  = k[..., :64]          # [N, 2, 64]
        k_pass = k[..., 64:]          # [N, 2, 192]
        v = v_proj(hidden)           # [N, 512]  = [2*256]
        kv_concat = cat(k_rot_flat, k_pass_flat, v_flat)  # [N, 1024]
        out = kv_a_proj(kv_concat)   # [N, 576]  = [latent(512), rope(64)]
    """

    def __init__(
        self,
        hidden_size: int,
        num_kv_heads: int,
        head_dim: int,
        rope_dim: int,
        kv_lora_rank: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.kv_lora_rank = kv_lora_rank

        self.nope_dim = head_dim - rope_dim  # 192
        self.k_proj = ReplicatedLinear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            prefix="k_proj",
        )
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self.v_proj = ReplicatedLinear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            prefix="v_proj",
        )

        kv_input_dim = (
            num_kv_heads * rope_dim       # k_rot  = 128
            + num_kv_heads * self.nope_dim  # k_pass = 384
            + num_kv_heads * head_dim       # v      = 512
        )                                   # total  = 1024
        self.kv_a_proj = nn.Linear(kv_input_dim, kv_lora_rank + rope_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        bs_seq = hidden_states.shape[:-1]

        # --- K path ---
        k = self.k_proj(hidden_states)[0]                       # [N, 512]
        k = k.view(*bs_seq, self.num_kv_heads, self.head_dim)   # [N, 2, 256]
        k = self.k_norm(k)                                       # RMSNorm on 256
        k_rot = k[..., : self.rope_dim]                          # [N, 2, 64]
        k_pass = k[..., self.rope_dim :]                         # [N, 2, 192]

        # --- V path ---
        v = self.v_proj(hidden_states)[0]                       # [N, 512]
        v = v.view(*bs_seq, self.num_kv_heads, self.head_dim)   # [N, 2, 256]

        # --- flatten & concat ---
        k_rot_flat = k_rot.reshape(*bs_seq, -1)                 # [N, 128]
        k_pass_flat = k_pass.reshape(*bs_seq, -1)               # [N, 384]
        v_flat = v.reshape(*bs_seq, -1)                         # [N, 512]

        kv_concat = torch.cat([k_rot_flat, k_pass_flat, v_flat], dim=-1)  # [N, 1024]
        out = self.kv_a_proj(kv_concat)                          # [N, 576]
        return (out,)


class Qwen3_5QProj(nn.Module):
    """Q projection that also carries an output gate.

    The checkpoint stores a single ``q_proj.weight`` of shape
    ``[num_heads * head_dim * 2, hidden_size]`` which we split into
    query (first half, qk_head_dim per head) and gate (second half,
    head_dim per head).

    Uses a single ``ColumnParallelLinear`` for the merged q+gate weight.
    The TP sharding correctly splits by head because the weight layout
    is [head0_q, head0_g, head1_q, head1_g, ...] interleaved per head.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        qk_head_dim: int,
        rms_norm_eps: float,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_head_dim = qk_head_dim  # 256 = nope(192) + rope(64)

        # Interleaved layout: [head0_q(256), head0_g(256), head1_q(256), ...]
        # This ensures TP sharding by dim 0 gives each rank the correct
        # set of heads with both query and gate parts.
        # Total output = num_heads * (qk_head_dim + head_dim) = 16 * 512 = 8192
        self.qg_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * (qk_head_dim + head_dim),  # 8192
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qg_proj",
        )
        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self._qk_head_dim = qk_head_dim
        self._head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        q_and_gate = self.qg_proj(hidden_states)[0]
        # Output is interleaved: [h0_q(256), h0_g(256), h1_q(256), h1_g(256), ...]
        # After TP sharding: [h0_q, h0_g, h1_q, h1_g, ...] for local heads
        # Each head's block = qk_head_dim + head_dim = 512
        block = self._qk_head_dim + self._head_dim  # 512
        local_heads = q_and_gate.shape[-1] // block
        # Reshape to [tokens, local_heads, block] then split q/g
        qg = q_and_gate.view(-1, local_heads, block)
        query = qg[..., : self._qk_head_dim]  # [tokens, heads, qk_head_dim]
        # gate = qg[..., self._qk_head_dim:]  # not needed here
        query = query.view(-1, local_heads, self.head_dim)
        query = self.q_norm(query)
        return (
            query.reshape(-1, local_heads * self.qk_head_dim),
        )

    def get_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the gate tensor [N, local_heads * head_dim]."""
        q_and_gate = self.qg_proj(hidden_states)[0]
        block = self._qk_head_dim + self._head_dim  # 512
        local_heads = q_and_gate.shape[-1] // block
        qg = q_and_gate.view(-1, local_heads, block)
        gate = qg[..., self._qk_head_dim:]  # [tokens, heads, head_dim]
        return gate.reshape(-1, local_heads * self._head_dim)


# ════════════════════════════════════════════════════════════════════
#  2.  Custom MLA Impl
# ════════════════════════════════════════════════════════════════════


class Qwen3_5MLAImplMixin:
    """Mixin that overrides AscendMLAImpl methods for Qwen3.6.

    This is mixed with :class:`AscendMLAImpl` at runtime (inside
    ``Qwen3_5MLAAttention.__init__``) to avoid the circular import caused by
    importing ``vllm_ascend.attention.mla_v1`` at module level.

    Overrides:
        * ``exec_kv_decode`` / ``exec_kv_prefill`` — skip
          ``npu_kv_rmsnorm_rope_cache`` (no latent RMSNorm), do RoPE +
          paged-cache scatter in Python.
        * ``process_weights_after_loading`` — extract ``W_UK_T`` / ``W_UV``
          from the per-kv-head ``kv_b_proj``, broadcasting to all query-heads.
        * ``forward`` — apply the output gate (``sigmoid(gate)``) before o_proj.
    """

    @staticmethod
    def _rotate_half(x):
        """GPT-NeoX style rotate_half, matching the original Qwen3.6 model."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope_neox(self, x, cos, sin):
        """Apply GPT-NeoX style RoPE using cos/sin from attn_metadata."""
        rd = self.qk_rope_head_dim
        cos_rd = cos[..., :rd]
        sin_rd = sin[..., :rd]

        orig_shape = x.shape
        if x.dim() == 3:
            x = x.unsqueeze(2)

        if cos_rd.dim() == 4:
            cos_rd = cos_rd[: x.shape[0]]
            sin_rd = sin_rd[: x.shape[0]]

        out = x * cos_rd + self._rotate_half(x) * sin_rd

        if len(orig_shape) == 3:
            out = out.squeeze(2)
        return out

    # ── cache write (decode) ──────────────────────────────────────────
    def exec_kv_decode(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads  # 1 for MLA
        S = 1
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)

        # Split latent and rope (NO RMSNorm — model has no latent norm)
        kv_c, k_pe = kv_no_split.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # RoPE on k_pe — use GPT-NeoX style (rotate_half), NOT npu_interleave_rope
        k_pe = self._apply_rope_neox(k_pe, cos, sin)

        # Paged-cache scatter
        block_size = kv_cache[0].shape[1]
        block_idx = (slots // block_size).long()
        block_off = (slots % block_size).long()
        kv_c_sq = kv_c.squeeze(2)   # [B, N, kv_lora_rank]
        k_pe_sq = k_pe.squeeze(2)  # [B, N, rope_dim]
        kv_cache[0][block_idx, block_off] = kv_c_sq
        kv_cache[1][block_idx, block_off] = k_pe_sq

        # Return the full cache tensors (not single-token slices) so that
        # _forward_decode can view them as [num_blocks, block_size, num_kv_heads, dim].
        return kv_cache[1], kv_cache[0]

    # ── cache write (prefill) ─────────────────────────────────────────
    def exec_kv_prefill(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads  # 1 for MLA
        S = 1
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)

        kv_c, k_pe = kv_no_split.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # RoPE on k_pe — use GPT-NeoX style (rotate_half)
        k_pe = self._apply_rope_neox(k_pe, cos, sin)

        block_size = kv_cache[0].shape[1]
        block_idx = (slots // block_size).long()
        block_off = (slots % block_size).long()
        kv_c_sq = kv_c.squeeze(2)
        k_pe_sq = k_pe.squeeze(2)
        kv_cache[0][block_idx, block_off] = kv_c_sq
        kv_cache[1][block_idx, block_off] = k_pe_sq

        # Return only the prefill tokens' values (not the full cache).
        # mla_preprocess_prefill passes k_nope (=kv_c) to kv_b_proj, which
        # should only process the prefill tokens, not the entire cache.
        return k_pe, kv_c

    # ── weight absorption setup ───────────────────────────────────────
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)

        # kv_b_proj.weight: [num_kv_orig*(nope+v), kv_lora_rank]
        # e.g. [4*(192+256), 512] = [1792, 512] for 4 kv_heads
        # Layout: [k0_pass(192), k1_pass(192), ..., v0(256), v1(256), ...] stacked per kv_head
        kv_b_proj_weight = self.kv_b_proj.weight.data.t().contiguous()  # [512, 896]
        num_kv_orig = self.num_kv_heads_original  # 2
        expected = num_kv_orig * (self.qk_nope_head_dim + self.v_head_dim)
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            expected,
        ), f"{kv_b_proj_weight.shape=}, expected ({self.kv_lora_rank}, {expected})"

        # Split into K and V parts FIRST (they are stacked, not interleaved per head)
        # [k0(192), k1(192), ..., v0(256), v1(256), ...]
        W_UK_all = kv_b_proj_weight[:, : num_kv_orig * self.qk_nope_head_dim]  # [512, 384]
        W_UV_all = kv_b_proj_weight[:, num_kv_orig * self.qk_nope_head_dim :]  # [512, 512]
        # Reshape per kv_head
        W_UK = W_UK_all.view(self.kv_lora_rank, num_kv_orig, self.qk_nope_head_dim)  # [512, 2, 192]
        W_UV = W_UV_all.view(self.kv_lora_rank, num_kv_orig, self.v_head_dim)  # [512, 2, 256]

        # Expand per-kv-head weights to per-query-head.
        # GQA grouping: kv_head g serves query heads [g*group_size:(g+1)*group_size]
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        group_size = self.num_heads * tp_size // num_kv_orig  # e.g. 16//4 = 4
        start_head = tp_rank * self.num_heads  # first local head index (global)
        # Each local head maps to a kv_head: kv_head = (start_head + local_idx) // group_size
        # Build per-local-head weights by selecting the correct kv_head for each.
        W_UK_expanded = []
        W_UV_expanded = []
        for h in range(self.num_heads):
            global_head = start_head + h
            kv_idx = global_head // group_size
            W_UK_expanded.append(W_UK[:, kv_idx, :])  # [512, 192]
            W_UV_expanded.append(W_UV[:, kv_idx, :])  # [512, 256]
        W_UK = torch.stack(W_UK_expanded, dim=1)  # [512, num_heads_local, 192]
        W_UV = torch.stack(W_UV_expanded, dim=1)  # [512, num_heads_local, 256]

        # Store in the layout expected by AscendMLAImpl (decode absorption)
        self.W_UV = W_UV.transpose(0, 1).contiguous()        # (8, 512, 256)
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()      # (8, 192, 512)
        self.W_UK_T = maybe_trans_nz(self.W_UK_T)

        # Rebuild kv_b_proj weight expanded to num_heads for prefill path.
        kv_b_proj_expanded = torch.cat(
            [W_UK, W_UV], dim=-1
        )  # [512, 8, 448]
        kv_b_proj_expanded = kv_b_proj_expanded.permute(1, 2, 0).reshape(
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            self.kv_lora_rank,
        ).contiguous()  # [3584, 512]
        # Replace the parameter with expanded version
        self.kv_b_proj.weight = nn.Parameter(kv_b_proj_expanded, requires_grad=False)

        del kv_b_proj_weight, W_UK, W_UV
        torch.npu.empty_cache()

    # ── Q/K split with RoPE-first layout (Qwen3 style) ────────────────
    # In Qwen3, the RoPE part is the FIRST qk_rope_head_dim dims of each head,
    # not the last. DeepSeek MLA assumes RoPE is last.  We override the
    # methods that split Q/K into nope/pe parts to use the Qwen3 layout.

    def rope_single(self, x, cos, sin):
        """Override to use GPT-NeoX style RoPE instead of npu_interleave_rope."""
        B, N, D = x.shape
        return self._apply_rope_neox(x, cos, sin).view(B, N, D)

    def _q_proj_and_k_up_proj(self, x):
        q = self.q_proj(x)[0].view(-1, self.num_heads, self.qk_head_dim)
        # Qwen3 layout: q[:qk_rope_head_dim] = rope, q[qk_rope_head_dim:] = nope
        q_pe = q[..., : self.qk_rope_head_dim]
        q_nope = q[..., self.qk_rope_head_dim :]
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def mla_preprocess_prefill(self, q_c, kv_no_split, kv_cache, attn_metadata):
        from vllm_ascend.attention.mla_v1 import PrefillMLAPreprocessResult
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        prefill_kv_no_split = kv_no_split[num_decode_tokens:num_actual_tokens]
        prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
        prefill_q = self.q_proj(prefill_q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
        # Qwen3 layout: rope first, nope second
        prefill_q_pe = prefill_q[..., : self.qk_rope_head_dim]
        prefill_q_nope = prefill_q[..., self.qk_rope_head_dim :]
        cos = attn_metadata.prefill.cos
        sin = attn_metadata.prefill.sin
        prefill_slots = attn_metadata.slot_mapping[num_decode_tokens:num_actual_tokens]
        prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
        prefill_k_pe, prefill_k_c_normed = self.exec_kv_prefill(prefill_kv_no_split, cos, sin, kv_cache, prefill_slots)
        prefill_k_nope, prefill_value = (
            self.kv_b_proj(prefill_k_c_normed)[0]
            .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        )
        prefill_k_pe = prefill_k_pe.view(prefill_q_c.shape[0], self.num_kv_heads, -1)
        prefill_k_pe = prefill_k_pe.expand((*prefill_k_nope.shape[:-1], -1))

        return PrefillMLAPreprocessResult(prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe, prefill_value)

    # ── override _compute_prefill_context to force cat(q_pe, q_nope) ──
    # NPU FIA requires query head_dim >= value head_dim (256).
    # When head_padding==0, the parent uses query=q_nope (192) which is < 256.
    # We force query=cat(q_pe, q_nope)=256 and key=cat(k_pe, k_nope)=256.
    def _compute_prefill_context(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: tuple[torch.Tensor],
        rope_dim: int,
        attn_metadata,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
    ):
        assert len(kv_c_and_k_pe_cache) > 1
        prefill_metadata = attn_metadata.prefill
        if prefill_metadata is None or prefill_metadata.chunked_context is None:
            return prefix_output, prefix_lse

        from vllm_ascend.device.device_op import DeviceOperator

        iters = len(prefill_metadata.chunked_context.seq_tot)
        cache_kv_c = kv_c_and_k_pe_cache[0]
        cache_k_pe = kv_c_and_k_pe_cache[1]
        num_heads = cache_k_pe.size(2)
        latent_kv_dim = kv_c_and_k_pe_cache[0].size(-1)

        actual_seq_lengths_q = prefill_metadata.actual_seq_lengths_q

        if iters == 0:
            return prefix_output, prefix_lse

        num_tokens = q_nope.size(0)
        D = self.v_head_dim
        H = self.num_heads

        if prefix_lse.dim() == 2:
            prefix_lse = prefix_lse.transpose(0, 1).unsqueeze(-1)
        prefix_output = prefix_output.to(torch.float32)
        prefix_lse = prefix_lse.to(torch.float32)
        out_list = [prefix_output.reshape(num_tokens * H, D)]
        lse_list = [prefix_lse.reshape(num_tokens * H)]

        # Force concatenated query (q_pe + q_nope = 256 >= v_head_dim 256)
        query = torch.cat((q_pe, q_nope), dim=-1)

        common_kwargs = {
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_heads,
            "input_layout": "TND",
            "atten_mask": None,
            "sparse_mode": 0,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "softmax_lse_flag": True,
            "actual_seq_lengths": actual_seq_lengths_q,
        }

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            context_seq_len_npu = self.get_context_seq_len_npu(i, attn_metadata)
            kv_c_normed = torch.empty(toks, num_heads, latent_kv_dim, dtype=cache_kv_c.dtype, device=cache_kv_c.device)
            k_pe = torch.empty(toks, num_heads, rope_dim, dtype=q_pe.dtype, device=q_pe.device)

            DeviceOperator.kv_cache_load(
                cache_kv_c,
                cache_k_pe,
                prefill_metadata.block_table,
                context_seq_len_npu,
                prefill_metadata.chunked_context.starts[i],
                key=kv_c_normed,
                value=k_pe,
            )

            kv_c_normed, k_pe = self._reorg_kvcache(
                kv_c_normed,
                k_pe,
                chunked_context=prefill_metadata.chunked_context,
                chunk_idx=i,
                toks=toks,
            )
            kv_c_normed = kv_c_normed.squeeze()
            if self.fa_quant_layer and get_ascend_device_type() == AscendDeviceType.A5:
                kv_c_normed = torch.mul(kv_c_normed.to(self.fak_descale_float.dtype), self.fak_descale_float).to(
                    torch.bfloat16
                )
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = k_pe.expand((*k_nope.shape[:-1], -1))

            actual_seq_lengths_kv = prefill_metadata.chunked_context.chunk_actual_seq_lengths_kv_list[i]
            common_kwargs["actual_seq_lengths_kv"] = actual_seq_lengths_kv

            # Force concatenated key (k_pe + k_nope = 256)
            key = torch.cat((k_pe, k_nope), dim=-1)

            chunk_out, chunk_lse = torch_npu.npu_fused_infer_attention_score(
                query, key.contiguous(), v.contiguous(), **common_kwargs
            )

            if chunk_lse.dim() == 2:
                chunk_lse = chunk_lse.transpose(0, 1).unsqueeze(-1)
            chunk_out = chunk_out.to(torch.float32)
            chunk_lse = chunk_lse.to(torch.float32)
            out_list.append(chunk_out.reshape(num_tokens * H, D))
            lse_list.append(chunk_lse.reshape(num_tokens * H))

        output_final, _ = torch_npu.npu_attention_update(tuple(lse_list), tuple(out_list), 0)
        return output_final.view(num_tokens, H, D), None

    # ── prefill attention (force concatenated q/k) ────────────────────
    def _forward_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: tuple[torch.Tensor],
        attn_metadata,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None
        assert len(kv_c_and_k_pe_cache) > 1
        num_tokens = q_nope.size(0)
        prefill_meta = attn_metadata.prefill

        actual_seq_lengths_q = prefill_meta.actual_seq_lengths_q
        actual_seq_lengths_kv = actual_seq_lengths_q.copy()

        original_dtype = q_nope.dtype
        need_dtype_convert = original_dtype != torch.bfloat16
        if need_dtype_convert:
            q_nope = q_nope.to(torch.bfloat16)
            q_pe = q_pe.to(torch.bfloat16)
            k_nope = k_nope.to(torch.bfloat16)
            k_pe = k_pe.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        attn_output = torch.empty(
            num_tokens, self.num_heads, self.v_head_dim,
            dtype=q_nope.dtype, device=q_nope.device,
        )
        attn_lse = torch.empty(
            self.num_heads, num_tokens, dtype=torch.float32, device=q_nope.device,
        )

        common_kwargs = {
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_heads,
            "input_layout": "TND",
            "atten_mask": prefill_meta.attn_mask,
            "sparse_mode": 3,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "block_table": None,
            "block_size": 0,
            "softmax_lse_flag": True,
            "actual_seq_lengths": actual_seq_lengths_q,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
        }
        record_attention_compute_start()

        # Always concatenate q_pe+q_nope and k_pe+k_nope (RoPE first, matching
        # original Qwen3.6 model: [rot(64), nope(192)] = 256 >= v_head_dim(256))
        query = torch.cat((q_pe, q_nope), dim=-1)
        key = torch.cat((k_pe, k_nope), dim=-1)

        attn_output, attn_lse = torch_npu.npu_fused_infer_attention_score(
            query, key.contiguous(), value.contiguous(), **common_kwargs
        )

        attn_output, attn_lse = self._compute_prefill_context(
            q_nope, q_pe, kv_c_and_k_pe_cache, self.qk_rope_head_dim,
            attn_metadata, attn_output, attn_lse,
        )

        attn_output = attn_output.reshape(
            [num_tokens, self.num_heads * self.v_head_dim],
        )

        if need_dtype_convert:
            attn_output = attn_output.to(original_dtype)

        return attn_output

    # ── forward with gate ─────────────────────────────────────────────
    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = self.get_num_actual_tokens(attn_metadata)
        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        num_decode_tokens = attn_metadata.num_decode_tokens
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        o_proj_input_shape = (
            _EXTRA_CTX.num_tokens,
            self.num_heads * self.v_head_dim,
        )
        o_proj_input = torch.zeros(
            o_proj_input_shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # ── Compute gate ──
        hs_gathered = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states.contiguous(), need_gather_q_kv
        )
        gate = self.q_proj.get_gate(hs_gathered[:num_actual_tokens])

        # ── MLA Preprocess ──
        decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess(
            layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv
        )

        if decode_preprocess_res is not None:
            output_decode = self._forward_decode(
                decode_preprocess_res.ql_nope,
                decode_preprocess_res.q_pe,
                decode_preprocess_res.k_nope,
                decode_preprocess_res.k_pe,
                kv_cache[0].shape[1],
                attn_metadata,
                getattr(decode_preprocess_res, "dequant_scale_q_nope", None),
            )
            o_proj_input[:num_decode_tokens] = output_decode

        if prefill_preprocess_res is not None:
            output_prefill = self._forward_prefill(
                prefill_preprocess_res.q_nope,
                prefill_preprocess_res.q_pe,
                prefill_preprocess_res.k_nope,
                prefill_preprocess_res.k_pe,
                prefill_preprocess_res.value,
                kv_cache,
                attn_metadata,
            )
            o_proj_input[num_decode_tokens:num_actual_tokens] = output_prefill

        # ── Apply gate before o_proj ──
        gate = gate.view(num_actual_tokens, self.num_heads * self.v_head_dim)
        o_proj_input[:num_actual_tokens] = (
            o_proj_input[:num_actual_tokens] * torch.sigmoid(gate)
        )

        # ── O proj ──
        output[...] = self.o_proj(
            o_proj_input, is_prefill=prefill_preprocess_res is not None
        )[0]

        del o_proj_input
        from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output


# ════════════════════════════════════════════════════════════════════
#  2b.  DSA indexer module  (learned wq_b / wk / weights_proj / k_norm)
# ════════════════════════════════════════════════════════════════════


class Qwen3_5Indexer(nn.Module):
    """Lightweight DSA indexer for Qwen3.5-MLA-DSA.

    Structurally identical to DeepSeek-V3.2's ``Indexer``
    (``vllm.model_executor.models.deepseek_v2.Indexer``): learned ``wq_b`` /
    fused ``wk_weights_proj`` / ``k_norm`` + a K-only paged cache, producing
    top-k key indices consumed by the SFA sparse-attention kernel.

    The single difference from DeepSeek-V3.2: ``wq_b`` takes ``hidden_states``
    directly (Qwen3.5 has no Q-lora residual), so its input dim is
    ``hidden_size`` instead of ``q_lora_rank``.  The indexer forward itself is
    driven by ``AscendSFAImpl.indexer_select_pre_process`` /
    ``indexer_select_post_process`` (which call ``wq_b`` / ``wk_weights_proj``
    / ``k_norm`` directly), so this class only owns the weights + cache.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        quant_config,
        cache_config,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        # Qwen3.5 has no Q-lora; wq_b consumes hidden_states directly.
        self.q_lora_rank = hidden_size
        self.softmax_scale = self.head_dim ** -0.5

        # No tensor parallel — replicated.
        self.wq_b = ReplicatedLinear(
            hidden_size,                         # <-- hidden, not q_lora
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # The checkpoint stores them separately (indexer.wk, indexer.weights_proj);
        # load_weights concatenates them into this fused weight.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.topk_indices_buffer = topk_indices_buffer

        # K-only paged cache.  The model creates it here; the runner overrides
        # the allocation with an AscendSFAIndexerCacheSpec (bf16,
        # head_size=index_head_dim on 910B3 non-C8).  The prefix must be unique
        # and registered so model_runner detects it via isinstance(...,
        # DeepseekV32IndexerCache).
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.prefix = prefix

    def forward(self):
        # The indexer compute is driven by AscendSFAImpl's
        # indexer_select_pre_process / indexer_select_post_process; this stub
        # exists only so nn.Module treats the weights as submodules.
        return


# ════════════════════════════════════════════════════════════════════
#  2c.  DSA impl overrides (Qwen MLA preprocess + SFA indexer/sparse attn)

# ════════════════════════════════════════════════════════════════════


class Qwen3_5DSAImplMixin:
    """DSA forward overrides for AscendSFAImpl tailored to Qwen3.5 MLA.

    Mixed as ``Qwen3_5DSAImpl(Qwen3_5DSAImplMixin, Qwen3_5MLAImplMixin,
    AscendSFAImpl)``.  The SFA base ``forward`` assumes DeepSeek's
    ``fused_qkv_a_proj`` / ``q_a_layernorm`` (which Qwen3.5 lacks, and which
    would assert-fail), so this provides a self-contained ``forward`` that:

    * runs the Qwen MLA preprocess — ``kv_a_proj_with_mqa`` (-> [latent, rope]),
      ``q_proj`` + gate, NeoX ``rotate_half`` RoPE, GQA weight absorption
      (all via the copied ``Qwen3_5MLAImplMixin`` helpers);
    * delegates the indexer top-k selection + sparse MLA attention to the
      reusable ``AscendSFAImpl`` helpers (``indexer_select_pre_process`` /
      ``indexer_select_post_process`` / ``_execute_sparse_flash_attention_process``
      / ``_v_up_proj`` / ``_compose_sfa_kv_cache``).

    Note on the indexer's ``wq_b`` input: DeepSeek-V3.2 feeds the Q-lora
    latent ``q_c`` into ``wq_b``; Qwen3.5 has no Q-lora, so ``wq_b`` takes
    ``hidden_states``.  ``AscendSFAImpl.indexer_select_post_process`` runs
    ``self.wq_b(q_c)`` — we simply pass ``q_c=hidden_states`` when calling it,
    so no override of that method is needed.
    """

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            return output.fill_(0)

        # Compose (k_cache[latent], v_cache[k_pe], indexer_k_cache[index_k]).
        composed_kv_cache = self._compose_sfa_kv_cache(kv_cache)
        assert composed_kv_cache is not None, (
            f"SFA kv cache not composed for layer_name={self.layer_name}")
        kv_cache = composed_kv_cache

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        actual_seq_lengths_query = attn_metadata.cum_query_lens
        actual_seq_lengths_key = attn_metadata.seq_lens
        num_input_tokens = attn_metadata.num_input_tokens

        # TP gather (same pattern as the plain Qwen MLA forward).
        hs = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states.contiguous(), need_gather_q_kv
        )

        # ── Qwen gate (applied via sigmoid before o_proj) ──
        gate = self.q_proj.get_gate(hs)

        # ── Qwen kv_a_proj_with_mqa -> [latent(512), rope(64)] ──
        kv_no_split = self.kv_a_proj_with_mqa(hs)[0]

        # ── Indexer K (reusable AscendSFAImpl helper; uses wk/k_norm) ──
        if self.has_indexer:
            k_li, _ = self.indexer_select_pre_process(x=hs, cos=cos, sin=sin)
        else:
            k_li = None

        # ── Qwen exec_kv: scatter latent->kv_cache[0], k_pe->kv_cache[1] ──
        self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata)

        # ── Qwen q_proj + k_up_proj (RoPE-first, from Qwen3_5MLAImplMixin) ──
        ql_nope, q_pe = self._q_proj_and_k_up_proj(hs)
        q_pe = self.rope_single(q_pe, cos, sin)

        # ── Scatter indexer K into kv_cache[2] (non-C8: dsa_k_cache_idx=2) ──
        if self.has_indexer and k_li is not None:
            dsa_k_cache_idx = 2
            torch_npu.npu_scatter_nd_update_(
                kv_cache[dsa_k_cache_idx].view(-1, k_li.shape[-1]),
                slot_mapping.view(-1, 1),
                k_li.view(-1, k_li.shape[-1]),
            )
            from vllm_ascend.attention.utils import notify_kv_cache_written
            notify_kv_cache_written(self.layer_name or "")

        # ── Top-k selection (reusable AscendSFAImpl helper) ──
        # Pass q_c=hs so wq_b consumes hidden_states (Qwen has no Q-lora).
        if self.skip_topk:
            topk_indices = self._get_indexcache_topk_indices(num_input_tokens)
        else:
            topk_indices = self.indexer_select_post_process(
                x=hs,
                q_c=hs,  # <-- wq_b(hidden_states) instead of wq_b(q_lora_latent)
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                cos=cos,
                sin=sin,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )
            if self.use_index_cache:
                self._update_indexcache_topk_indices(topk_indices)

        # ── Sparse MLA attention (reusable AscendSFAImpl helper) ──
        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

        # ── V up-projection (W_UV) ──
        attn_output = self._v_up_proj(attn_output)

        # ── Apply Qwen gate before o_proj ──
        attn_output = attn_output * torch.sigmoid(gate)

        # ── O proj ──
        output[...] = self.o_proj(attn_output)[0]

        from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        return output

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata,
    ):
        """Qwen KV cache write (no latent RMSNorm, NeoX RoPE on k_pe).

        Mirrors ``Qwen3_5MLAImplMixin.exec_kv_prefill`` but matches the SFA
        ``exec_kv`` signature and writes into the composed SFA cache tuple
        ``(kv_cache[0]=latent, kv_cache[1]=k_pe)``.  Returns ``(None, None)``
        like the SFA base for the non-CP/non-C8 path (the sparse-attention
        kernel reads from the paged cache directly).
        """
        B = kv_no_split.shape[0]
        N = self.num_kv_heads  # 1 for MLA
        S = 1
        kv_no_split = kv_no_split.view(
            B, N, S, self.kv_lora_rank + self.qk_rope_head_dim
        )
        kv_c, k_pe = kv_no_split.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # NeoX RoPE on k_pe (no latent RMSNorm — Qwen3.5 has kv_a_layernorm=Identity)
        k_pe = self._apply_rope_neox(k_pe, cos, sin)

        block_size = kv_cache[0].shape[1]
        block_idx = (slots // block_size).long()
        block_off = (slots % block_size).long()
        kv_c_sq = kv_c.squeeze(2)   # [B, N, kv_lora_rank]
        k_pe_sq = k_pe.squeeze(2)   # [B, N, rope_dim]
        kv_cache[0][block_idx, block_off] = kv_c_sq
        kv_cache[1][block_idx, block_off] = k_pe_sq
        return None, None


# ════════════════════════════════════════════════════════════════════
#  3.  Attention layer


# ════════════════════════════════════════════════════════════════════


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5DSAForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Top-level multimodal model: vision encoder + MLA+DSA-adapted language model.

    Inherits from Qwen3_5ForConditionalGeneration to reuse all weight loading,
    prefix mapping, multimodal processing, etc.  Only overrides
    language_model to use Qwen3_5DSAForCausalLM.
    """

    # Override mapper: compressed checkpoint uses model.xxx / lm_head.xxx
    # (not model.language_model.xxx like original Qwen3.5)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5DSAForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.embed_input_ids(input_ids)

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Combine prefix mapper (model. → language_model.model.) with model's
        # mapper (self_attn remapping + GDN stacking) into one, applied once.
        mapper = self.hf_to_vllm_mapper | self.language_model.model.hf_to_vllm_mapper
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights, mapper=mapper)
