# SPDX-License-Identifier: Apache-2.0
"""Patch Qwen3.5 MLA/DSA attention with Ascend-specific impl overrides.

The model architecture (submodules, MLAModules wiring, thin DecoderLayer/
Model/ForCausalLM subclasses) lives in vllm upstream and uses the generic
``MultiHeadLatentAttentionWrapper`` PluggableLayer.  On Ascend the wrapper
auto-dispatches to ``AscendMultiHeadLatentAttention`` which creates a default
``AscendMLAImpl`` (or ``AscendSFAImpl`` for DSA).  Qwen3.5's MLA differs from
DeepSeek's in several device-agnostic ways (NeoX RoPE, no latent norm, output
gate, GQA kv_b_proj broadcast) that are implemented in ``Qwen3_5MLAImplMixin``
and ``Qwen3_5DSAImplMixin`` (kept in ``vllm_ascend/models/``).

This patch wraps the upstream ``__init__`` to replace the default impl with
the Qwen3.5-specific mixin after the generic construction completes.
"""

from __future__ import annotations

import torch

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DSAAttention,
    Qwen3_5MLAAttention,
)

from vllm_ascend.models.qwen3_5_dsa import Qwen3_5DSAImplMixin
from vllm_ascend.models.qwen3_5_mla import Qwen3_5MLAImplMixin


def _patch_mla_attention() -> None:
    """Inject Qwen3_5MLAImplMixin into Qwen3_5MLAAttention on Ascend."""
    orig_init = Qwen3_5MLAAttention.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        from vllm_ascend.attention.mla_v1 import AscendMLAImpl

        class Qwen3_5MLAImpl(Qwen3_5MLAImplMixin, AscendMLAImpl):
            pass

        orig_impl = self.attn.mla_attn.impl
        custom_impl = Qwen3_5MLAImpl(
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
            layer_name=f"{self.prefix}.attn",
        )
        custom_impl.num_kv_heads_original = self.num_kv_heads_original
        self.attn.mla_attn.impl = custom_impl

        def _custom_process_weights(act_dtype: torch.dtype):
            custom_impl.process_weights_after_loading(act_dtype)

        self.attn.mla_attn.process_weights_after_loading = (
            _custom_process_weights
        )

    Qwen3_5MLAAttention.__init__ = patched_init


def _patch_dsa_attention() -> None:
    """Inject Qwen3_5DSAImplMixin into Qwen3_5DSAAttention on Ascend."""
    orig_init = Qwen3_5DSAAttention.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

        class Qwen3_5DSAImpl(
            Qwen3_5DSAImplMixin, Qwen3_5MLAImplMixin, AscendSFAImpl
        ):
            pass

        orig_impl = self.attn.mla_attn.impl
        custom_impl = Qwen3_5DSAImpl(
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
            indexer=orig_impl.indexer,
            skip_topk=orig_impl.skip_topk,
            topk_indices_buffer=orig_impl.topk_indices_buffer,
            layer_name=f"{self.prefix}.attn",
        )
        custom_impl.num_kv_heads_original = self.num_kv_heads_original
        self.attn.mla_attn.impl = custom_impl

        def _custom_process_weights(act_dtype: torch.dtype):
            custom_impl.process_weights_after_loading(act_dtype)

        self.attn.mla_attn.process_weights_after_loading = (
            _custom_process_weights
        )

    Qwen3_5DSAAttention.__init__ = patched_init


def apply_patches() -> None:
    _patch_mla_attention()
    _patch_dsa_attention()


apply_patches()
