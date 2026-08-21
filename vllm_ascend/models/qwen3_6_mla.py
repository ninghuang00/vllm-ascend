# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B MLA-adapted model for vLLM-Ascend.

This model adapts the Qwen3.6 KV-compressed architecture to vLLM-Ascend's
DeepSeek-style MLA attention path.  Key differences from standard DeepSeek MLA:

1. ``kv_a_proj_with_mqa`` wraps the original ``k_proj / k_norm / v_proj /
   kv_a_proj`` pipeline so that, externally, it looks like a single
   ``hidden -> [latent, rope]`` projection.
2. No ``kv_a_layernorm`` on the latent — the fused ``npu_kv_rmsnorm_rope_cache``
   operator is replaced by a Python-level ``npu_interleave_rope`` + paged-cache
   scatter.
3. ``kv_b_proj`` is stored per kv-head (2) and broadcast to all query-heads
   (16) during ``process_weights_after_loading`` to produce ``W_UK_T`` /
   ``W_UV`` for the weight-absorption decode path.
4. ``q_proj`` carries an output gate (``attn_output_gate``); the gate is split
   off and applied via ``sigmoid`` right before ``o_proj``.
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
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.layernorm import RMSNorm
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
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

# NOTE: vllm_ascend.attention.* imports are done lazily (inside methods) to
# avoid a circular-import chain: mla_v1 → attention_v1 → device_op →
# ops.fused_moe → experts_selector → device_op.
from vllm_ascend.memcache_comm_fence import record_attention_compute_start
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
from vllm_ascend.utils import maybe_trans_nz

from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeProcessingInfo,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
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


class Qwen3_6KVProjWithMQA(nn.Module):
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


class Qwen3_6QProj(nn.Module):
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


class Qwen3_6MLAImplMixin:
    """Mixin that overrides AscendMLAImpl methods for Qwen3.6.

    This is mixed with :class:`AscendMLAImpl` at runtime (inside
    ``Qwen3_6MLAAttention.__init__``) to avoid the circular import caused by
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

        # kv_b_proj.weight: [896, 512] = [num_kv_orig*(nope+v), kv_lora_rank]
        # Layout: [k0_pass(192), k1_pass(192), v0(256), v1(256), ...] stacked per kv_head
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
        group_size = self.num_heads * tp_size // num_kv_orig  # 16 // 2 = 8
        start_head = tp_rank * self.num_heads  # 0 or 8
        kv_head_idx = start_head // group_size  # 0 or 1
        # Select this rank's kv_head and expand to all local heads
        W_UK = W_UK[:, kv_head_idx:kv_head_idx + 1, :].expand(
            -1, self.num_heads, -1
        ).contiguous()  # [512, 8, 192]
        W_UV = W_UV[:, kv_head_idx:kv_head_idx + 1, :].expand(
            -1, self.num_heads, -1
        ).contiguous()  # [512, 8, 256]

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

        # DEBUG: dump prefill intermediate values for layer 3 only
        _is_l3 = "layers.3." in (self.layer_name or "")
        if _is_l3 and not getattr(self, '_dbg_l3', False) and prefill_q_c.shape[0] <= 128:
            self._dbg_l3 = True
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            for name, t in [
                ("q_c", prefill_q_c),
                ("q_pe_before_rope", prefill_q[..., : self.qk_rope_head_dim].contiguous()),
                ("q_pe_after_rope", prefill_q_pe.contiguous()),
                ("q_nope", prefill_q_nope.contiguous()),
                ("kv_no_split", prefill_kv_no_split.contiguous()),
                ("latent", prefill_k_c_normed.contiguous()),
                ("k_pe_after_rope", prefill_k_pe.contiguous()),
                ("k_nope", prefill_k_nope.contiguous()),
                ("value", prefill_value.contiguous()),
            ]:
                tc = t.detach().cpu().to(torch.float32)
                torch.save(tc, f"/tmp/vllm_l3_{name}_rank{rank}.pt")
            logger.warning("DUMPED layer3 prefill intermediate values rank=%d layer=%s", rank, self.layer_name)

        # DEBUG: dump all prefill intermediate values for layer 3 only
        if _is_l3 and not getattr(self, '_dbg_l3_full', False) and prefill_q_c.shape[0] <= 128:
            self._dbg_l3_full = True
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            # q_c (= hidden_states after input_layernorm)
            # q_proj output = self.q_proj(prefill_q_c) which returns (query_flat,)
            q_out = self.q_proj(prefill_q_c)[0]  # [N, local_heads*qk_head_dim]
            # gate = self.q_proj.get_gate(prefill_q_c)
            gate = self.q_proj.get_gate(prefill_q_c)  # [N, local_heads*head_dim]
            # q before norm: reshape to [N, local_heads, qk_head_dim]
            q_before_norm = q_out.view(-1, self.num_heads, self.qk_head_dim)
            # q after norm (what we actually use)
            # prefill_q is already after norm
            # k_proj, k_norm, v_proj outputs
            kv_a_out = prefill_kv_no_split  # [N, 576] = kv_a_proj_with_mqa output
            # latent and k_pe (before rope)
            # We need to re-derive these since exec_kv_prefill already applied rope
            # But we have kv_no_split, let's split manually
            kv_split = prefill_kv_no_split.view(prefill_kv_no_split.shape[0], self.num_kv_heads, 1, self.kv_lora_rank + self.qk_rope_head_dim)
            kv_c_manual, k_pe_manual = kv_split.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            
            for name, t in [
                ("q_c_input", prefill_q_c),
                ("q_proj_output", q_out),
                ("gate_output", gate),
                ("q_before_norm", q_before_norm),
                ("q_after_norm", prefill_q),  # [N, heads, qk_head_dim]
                ("q_pe_before_rope", prefill_q[..., :self.qk_rope_head_dim].contiguous()),
                ("q_pe_after_rope", prefill_q_pe.contiguous()),
                ("q_nope", prefill_q_nope.contiguous()),
                ("kv_no_split", prefill_kv_no_split.contiguous()),
                ("latent_before_rope", kv_c_manual.squeeze(2).contiguous()),
                ("k_pe_before_rope", k_pe_manual.squeeze(2).contiguous()),
                ("k_pe_after_rope", prefill_k_pe.contiguous()),
                ("k_nope", prefill_k_nope.contiguous()),
                ("value", prefill_value.contiguous()),
            ]:
                tc = t.detach().cpu().to(torch.float32)
                torch.save(tc, f"/tmp/vllm_l3d_{name}_rank{rank}.pt")
            logger.warning("DUMPED layer3 all prefill intermediates rank=%d layer=%s", rank, self.layer_name)

        return PrefillMLAPreprocessResult(prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe, prefill_value)

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

        # DEBUG: dump prefill attention output (layer 3 only)
        if "layers.3." in (self.layer_name or "") and not getattr(self, '_dbg_l3_attn', False) and num_tokens <= 128:
            self._dbg_l3_attn = True
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            for name, t in [
                ("query", query.contiguous()),
                ("key", key.contiguous()),
                ("attn_output", attn_output.contiguous()),
            ]:
                tc = t.detach().cpu().to(torch.float32)
                torch.save(tc, f"/tmp/vllm_l3_{name}_rank{rank}.pt")
            logger.warning("DUMPED layer3 prefill attn output rank=%d", rank)

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

        # DEBUG: dump o_proj output and final output (layer 3 only)
        if "layers.3." in (self.layer_name or "") and not getattr(self, '_dbg_l3_final', False) and num_actual_tokens <= 128:
            self._dbg_l3_final = True
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            tc = output.detach().cpu().to(torch.float32)
            torch.save(tc, f"/tmp/vllm_l3_attn_final_rank{rank}.pt")
            logger.warning("DUMPED layer3 attn final output rank=%d min=%.4f max=%.4f",
                           rank, output.min(), output.max())

        return output


# ════════════════════════════════════════════════════════════════════
#  3.  Attention layer
# ════════════════════════════════════════════════════════════════════


class Qwen3_6MLAAttention(nn.Module):
    """Full-attention layer using MLA for compressed-KV Qwen3.6."""

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        hidden_size = config.hidden_size           # 2048
        num_heads = config.num_attention_heads      # 16
        num_kv_heads = config.num_key_value_heads   # 2
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)  # 256

        # TP: local num_heads (for weight absorption and o_proj)
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
        tp_size = get_tensor_model_parallel_world_size()
        num_heads_local = num_heads // tp_size  # 8 when TP=2

        rope_params = config.rope_parameters or {}
        partial_rotary = rope_params.get("partial_rotary_factor", 1.0)
        rope_dim = int(head_dim * partial_rotary)  # 64
        nope_dim = head_dim - rope_dim             # 192

        kv_comp = getattr(config, "kv_compression", {}) or {}
        kv_lora_rank = kv_comp.get("rank", 512)

        # MLA dimension constants
        qk_rope_head_dim = rope_dim     # 64
        qk_nope_head_dim = nope_dim     # 192
        v_head_dim = head_dim            # 256
        q_lora_rank = None

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 256
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads_local
        self.num_kv_heads_original = num_kv_heads  # 2 (for weight broadcast)

        # ── Create modules ──
        self.kv_a_proj_with_mqa = Qwen3_6KVProjWithMQA(
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            kv_lora_rank=kv_lora_rank,
            rms_norm_eps=config.rms_norm_eps,
        )

        self.q_proj = Qwen3_6QProj(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_head_dim=self.qk_head_dim,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # No latent norm — identity placeholder for MLAModules
        self.kv_a_layernorm = nn.Identity()

        # kv_b_proj: output dim is per-kv-head (896); expanded to per-query-head
        # (7168) in process_weights_after_loading for prefill path compatibility.
        self.kv_b_proj = ReplicatedLinear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + v_head_dim),  # 896
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,  # FULL heads × v_head_dim (RowParallelLinear
            # shards input dim by TP internally, so pass the full count)
            hidden_size,              # 2048
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Rotary embedding — we use GPT-NeoX style (rotate_half) to match
        # the original Qwen3.6 model's RoPE implementation.
        # cos/sin are computed via the rotary_emb's cos_sin_cache, NOT via
        # npu_interleave_rope's _cos_cache (which is for interleaved format).
        max_pos = getattr(config, "max_position_embeddings", 32768)
        rope_params_clean = {
            k: v for k, v in rope_params.items()
            if k not in ("mrope_section", "mrope_interleaved", "partial_rotary_factor")
        }
        rope_params_clean["rope_type"] = rope_params_clean.get("rope_type", "default")
        rope_params_clean["partial_rotary_factor"] = 1.0
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_pos,
            rope_parameters=rope_params_clean,
            is_neox_style=True,
        )

        scale = head_dim ** -0.5

        # ── Build MLAModules ──
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=None,
            q_b_proj=None,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )

        # ── Create AscendMultiHeadLatentAttention ──
        # This will register as "multi_head_latent_attention" PluggableLayer
        self.attn = AscendMultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads_local,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # ── Replace impl with custom Qwen3_6MLAImpl ──
        # Lazy import to avoid circular import at module level
        from vllm_ascend.attention.mla_v1 import AscendMLAImpl

        class Qwen3_6MLAImpl(Qwen3_6MLAImplMixin, AscendMLAImpl):
            pass

        orig_impl = self.attn.mla_attn.impl
        custom_impl = Qwen3_6MLAImpl(
            num_heads=orig_impl.num_heads,
            head_size=orig_impl.head_size,
            scale=orig_impl.scale,
            num_kv_heads=orig_impl.num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=orig_impl.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type="DECODER",
            kv_sharing_target_layer_name=None,
            q_lora_rank=orig_impl.q_lora_rank,
            kv_lora_rank=orig_impl.kv_lora_rank,
            qk_nope_head_dim=orig_impl.qk_nope_head_dim,
            qk_rope_head_dim=orig_impl.qk_rope_head_dim,
            qk_head_dim=orig_impl.qk_head_dim,
            v_head_dim=orig_impl.v_head_dim,
            rotary_emb=self.rotary_emb,
            fused_qkv_a_proj=None,
            q_b_proj=None,
            q_a_layernorm=None,
            q_proj=self.q_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            layer_name=f"{prefix}.attn",
        )
        # Propagate num_kv_heads_original for weight broadcast
        custom_impl.num_kv_heads_original = num_kv_heads
        self.attn.mla_attn.impl = custom_impl

        # Override process_weights_after_loading on MLAAttention to skip
        # the parent's assert (which expects kv_b_proj output dim =
        # num_heads * (qk_nope + v_head), but our kv_b_proj is per-kv-head).
        def _custom_process_weights(act_dtype: torch.dtype):
            custom_impl.process_weights_after_loading(act_dtype)

        self.attn.mla_attn.process_weights_after_loading = _custom_process_weights

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(positions, hidden_states)


# ════════════════════════════════════════════════════════════════════
#  4.  Decoder layer
# ════════════════════════════════════════════════════════════════════


class Qwen3_6MLADecoderLayer(nn.Module):
    """Decoder layer: linear_attn layers unchanged, full_attn layers use MLA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
                QwenGatedDeltaNetAttention,
            )
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_6MLAAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # MLP
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        **kwargs,
    ):
        # DEBUG: dump layer 3 input
        if self.layer_idx in (1, 2, 3) and not getattr(self, f'_dbg_l{self.layer_idx}_in', False) and hidden_states.shape[0] <= 128:
            setattr(self, f'_dbg_l{self.layer_idx}_in', True)
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            torch.save(hidden_states.detach().cpu().to(torch.float32),
                       f"/tmp/vllm_l{self.layer_idx}_input_rank{rank}.pt")

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # DEBUG: dump residual and hidden_states before/after input_layernorm for layer 3
            if self.layer_idx == 3 and not getattr(self, '_dbg_l3_ln', False) and hidden_states.shape[0] <= 128:
                self._dbg_l3_ln = True
                from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
                rank = get_tensor_model_parallel_rank()
                torch.save(hidden_states.detach().cpu().to(torch.float32), f"/tmp/vllm_l3_ln_input_hs_rank{rank}.pt")
                torch.save(residual.detach().cpu().to(torch.float32), f"/tmp/vllm_l3_ln_input_res_rank{rank}.pt")
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
            if self.layer_idx == 3 and not getattr(self, '_dbg_l3_ln_out', False) and hidden_states.shape[0] <= 128:
                self._dbg_l3_ln_out = True
                from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
                rank = get_tensor_model_parallel_rank()
                torch.save(hidden_states.detach().cpu().to(torch.float32), f"/tmp/vllm_l3_ln_output_rank{rank}.pt")
                torch.save(residual.detach().cpu().to(torch.float32), f"/tmp/vllm_l3_ln_output_res_rank{rank}.pt")
                logger.warning("DUMPED layer3 input_layernorm: hs_in=%s res_in=%s hs_out=%s res_out=%s",
                              tuple(hidden_states.shape), tuple(residual.shape),
                              f"min={hidden_states.min():.4f} max={hidden_states.max():.4f}",
                              f"min={residual.min():.4f} max={residual.max():.4f}")

        stream = torch_npu.npu.current_stream()
        if self.layer_type == "full_attention":
            torch_npu.npu.mstx.mark("full_attention start", stream)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            torch_npu.npu.mstx.mark("full_attention end", stream)
        else:
            torch_npu.npu.mstx.mark("linear_attention start", stream)
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                **kwargs,
            )
            torch_npu.npu.mstx.mark("linear_attention end", stream)

            if self.layer_idx in (0, 1) and not getattr(self, f'_dbg_postgdn_l{self.layer_idx}', False) and hidden_states.shape[0] <= 128:
                setattr(self, f'_dbg_postgdn_l{self.layer_idx}', True)
                from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
                rank = get_tensor_model_parallel_rank()
                torch.save(hidden_states.detach().cpu().to(torch.float32),
                           f"/tmp/vllm_l{self.layer_idx}_postgdn_rank{rank}.pt")
                logger.warning("DUMPED layer0 postgdn rank=%s shape=%s min=%.4f max=%.4f",
                               rank, tuple(hidden_states.shape), hidden_states.min(), hidden_states.max())

        if self.layer_scale:
            hidden_states = hidden_states * (1.0 + self.attn_layer_scale)

        # For MLA, hidden_states is already the right shape
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        # DEBUG: dump MoE input/output for layer 0
        if self.layer_idx in (0, 1) and not getattr(self, f'_dbg_moe_l{self.layer_idx}', False) and hidden_states.shape[0] <= 128:
            setattr(self, f'_dbg_moe_l{self.layer_idx}', True)
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            torch.save(hidden_states.detach().cpu().to(torch.float32),
                       f"/tmp/vllm_l{self.layer_idx}_moe_input_rank{rank}.pt")
        
        hidden_states = self.mlp(hidden_states)
        
        if self.layer_idx in (0, 1) and not getattr(self, f'_dbg_moe_out_l{self.layer_idx}', False) and hidden_states.shape[0] <= 128:
            setattr(self, f'_dbg_moe_out_l{self.layer_idx}', True)
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            torch.save(hidden_states.detach().cpu().to(torch.float32),
                       f"/tmp/vllm_l{self.layer_idx}_moe_output_rank{rank}.pt")
            logger.warning("DUMPED layer%d moe output rank=%s min=%.4f max=%.4f norm=%.4f",
                           self.layer_idx, rank, hidden_states.min(), hidden_states.max(), hidden_states.norm())

        if self.layer_scale:
            hidden_states = hidden_states * (1.0 + self.ffn_layer_scale)

        # DEBUG: dump layer 2 full output (MLP + residual = what layer 3 receives)
        if self.layer_idx == 2 and not getattr(self, '_dbg_l2_full', False) and hidden_states.shape[0] <= 128:
            self._dbg_l2_full = True
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            full_out = (hidden_states + residual).detach().cpu().to(torch.float32)
            torch.save(full_out, f"/tmp/vllm_l2_full_output_rank{rank}.pt")
            logger.warning("DUMPED layer2 full output rank=%s min=%.4f max=%.4f norm=%.4f",
                          rank, full_out.min(), full_out.max(), full_out.norm())

        # DEBUG: dump layer 3 output (MLP output, before next layer's residual add)
        if self.layer_idx == 3 and not getattr(self, '_dbg_l3_out', False) and hidden_states.shape[0] <= 128:
            self._dbg_l3_out = True
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            torch.save(hidden_states.detach().cpu().to(torch.float32),
                       f"/tmp/vllm_l3_mlp_output_rank{rank}.pt")

        return hidden_states, residual


# ════════════════════════════════════════════════════════════════════
#  5.  Model
# ════════════════════════════════════════════════════════════════════


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_6MLAModel(Qwen3_5Model):
    """Qwen3.6 model with MLA-adapted full-attention layers."""

    # Custom mapper: do NOT inherit Qwen3_5Model's mapper (it has qkv stacking
    # that conflicts with our custom self_attn structure). Build from scratch:
    # 1. GDN in_proj fusing (in_proj_qkv+in_proj_z → in_proj_qkvz, etc.)
    # 2. MoE gate+up → gate_up_proj stacking
    # 3. self_attn weight remapping to kv_a_proj_with_mqa / q_proj submodules
    hf_to_vllm_mapper = (
        WeightsMapper(
            orig_to_new_stacked={
                ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
                ".in_proj_z": (".in_proj_qkvz", 3),
                ".in_proj_b": (".in_proj_ba", 0),
                ".in_proj_a": (".in_proj_ba", 1),
                ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
                ".mlp.up_proj": (".mlp.gate_up_proj", 1),
                ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
                ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
            }
        )
        | WeightsMapper(
            orig_to_new_substr={
                ".self_attn.k_proj.": ".self_attn.kv_a_proj_with_mqa.k_proj.",
                ".self_attn.k_norm.": ".self_attn.kv_a_proj_with_mqa.k_norm.",
                ".self_attn.v_proj.": ".self_attn.kv_a_proj_with_mqa.v_proj.",
                ".self_attn.kv_a_proj.": ".self_attn.kv_a_proj_with_mqa.kv_a_proj.",
                ".self_attn.q_norm.": ".self_attn.q_proj.q_norm.",
            }
        )
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Skip Qwen3_5Model.__init__ to avoid creating duplicate layers
        nn.Module.__init__(self)

        config: Qwen3_5MoeTextConfig = vllm_config.model_config.hf_text_config

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_6MLADecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        residual = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                hidden_states,
                residual,
                positions=positions,
            )

        if not get_pp_group().is_last_rank:
            return hidden_states

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


# ════════════════════════════════════════════════════════════════════
#  6.  ForCausalLM
# ════════════════════════════════════════════════════════════════════


class Qwen3_6MLAForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts, IsHybrid, SupportsMRoPE):
    """Language model wrapper.

    Inherits from Qwen3_5ForCausalLMBase to reuse all weight loading, mamba
    cache, LoRA, PP, Eagle3 support.  Mixes in QwenNextMixtureOfExperts for
    MoE expert TP slicing.  Only overrides model to use Qwen3_6MLAModel
    with MLA-adapted attention.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config

        self.model = Qwen3_6MLAModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters for expert TP slicing
        self.set_moe_parameters()

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features,
    ) -> tuple[torch.Tensor, int]:
        """Return M-RoPE positions for text-only input (3, seq_len)."""
        import numpy as np
        positions = np.arange(len(input_tokens))
        mrope_positions = np.broadcast_to(positions, (3, len(input_tokens)))
        return torch.tensor(mrope_positions), 0

    def set_moe_parameters(self):
        """Override to work with Qwen3_6MLADecoderLayer instead of Qwen3_5DecoderLayer."""
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint's q_proj.weight is already laid out as
        # [h0_q(256), h0_g(256), h1_q(256), h1_g(256), ...] (interleaved per head),
        # because the original Qwen3.6 model uses:
        #   q_proj(hidden) -> [seq, 8192]
        #   .view(seq, num_heads, head_dim*2)  # [seq, 16, 512]
        #   .chunk(2, dim=-1)  # q=[seq,16,256], gate=[seq,16,256]
        # So head h's query is at rows [h*512 : h*512+256] and gate at [h*512+256 : h*512+512].
        #
        # ColumnParallelLinear qg_proj expects exactly this interleaved layout
        # for correct TP sharding by head.  Just rename the weight.
        renamed_weights = []
        for name, data in weights:
            if name.endswith(".self_attn.q_proj.weight"):
                new_name = name.replace(".self_attn.q_proj.weight",
                                        ".self_attn.q_proj.qg_proj.weight")
                renamed_weights.append((new_name, data))
            else:
                renamed_weights.append((name, data))

        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(
            renamed_weights, mapper=self.model.hf_to_vllm_mapper
        )


# ════════════════════════════════════════════════════════════════════
#  7.  ForConditionalGeneration (multimodal)
# ════════════════════════════════════════════════════════════════════


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_6MLAForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Top-level multimodal model: vision encoder + MLA-adapted language model.

    Inherits from Qwen3_5ForConditionalGeneration to reuse all weight loading,
    prefix mapping, multimodal processing, etc.  Only overrides
    language_model to use Qwen3_6MLAForCausalLM.
    """

    is_3d_moe_weight: bool = True

    # Override mapper: compressed checkpoint uses model.xxx / lm_head.xxx
    # (not model.language_model.xxx like original Qwen3.6)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
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
            self.language_model = Qwen3_6MLAForCausalLM(
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
