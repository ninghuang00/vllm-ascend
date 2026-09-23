# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B MLA-adapted model for vLLM-Ascend.

This model adapts the Qwen3.5 KV-compressed architecture to vLLM-Ascend's
DeepSeek-style MLA attention path.  Key differences from standard DeepSeek MLA:

1. ``kv_a_proj_with_mqa`` wraps the original ``k_proj / k_norm / v_proj /
   kv_a_proj`` pipeline so that, externally, it looks like a single
   ``hidden -> [latent, rope]`` projection.
2. No ``kv_a_layernorm`` on the latent — the fused ``npu_kv_rmsnorm_rope_cache``
   operator is replaced by a Python-level ``npu_interleave_rope`` + paged-cache
   scatter.
3. ``kv_b_proj`` is stored per kv-head (4) and broadcast to all query-heads
   (16) during ``process_weights_after_loading`` to produce ``W_UK_T`` /
   ``W_UV`` for the weight-absorption decode path.
4. ``q_proj`` carries an output gate (``attn_output_gate``); the gate is split
   off and applied via ``sigmoid`` right before ``o_proj``.

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
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MLAForCausalLM,
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
#  3.  Attention layer


# ════════════════════════════════════════════════════════════════════


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MLAForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Top-level multimodal model: vision encoder + MLA-adapted language model.

    Inherits from Qwen3_5ForConditionalGeneration to reuse all weight loading,
    prefix mapping, multimodal processing, etc.  Only overrides
    language_model to use Qwen3_5MLAForCausalLM.
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
            self.language_model = Qwen3_5MLAForCausalLM(
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
