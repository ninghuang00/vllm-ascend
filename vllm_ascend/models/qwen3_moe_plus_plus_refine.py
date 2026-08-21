# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from qwen3_moe_plus_plus.py.
#
# Qwen3 MoE++ Refine variant:
#   1. softmax over 192 experts → top-8 (含零专家, 不mask)
#   2. 对这8个排序：真实专家(id < 128)在前，零专家在后
#   3. 取前 K_refine = K-2 = 6 个做 GEMM
#   4. 归一化用全部8个的权重和（保留零专家稀释效应）
#
#   实现方式：重写 apply()，在 select_experts 做 top-8 后，
#   用纯 tensor 操作排序+截取到 6，传给 GEMM pipeline。
#   FusedMoE 配置 top_k=6 保证 CANN 算子 active_num=N*6 兼容。

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.quantization.quant_type import QuantType


class RefineFusedMoEMethod(AscendUnquantizedFusedMoEMethod):
    """Ascend FusedMoE method with top-8 → sort → take K-2 routing.

    select_experts does top-8 over all 192 experts (including zero experts).
    Then we sort the 8 selected so zero experts move to the end, take first 6,
    with weights renormalized over all 8 (preserving dilution).
    """

    def apply(
        self,
        layer,
        x,
        use_grouped_topk,
        top_k,
        router_logits,
        renormalize,
        **kwargs,
    ):
        # top_k here is K_refine=6 (from moe_config), but we select top-8 first.
        top_k_full = 8  # K=8, select from all 192 experts

        num_experts = kwargs.get("num_experts", -1)
        input_ids = kwargs.get("input_ids")

        # Step 1: select_experts does softmax(192) + top-8 + renormalize=True.
        # This gives us 8 experts with weights summing to 1 (over 8).
        # num_experts=128 matches original behavior (only affects NPU fusion check).
        topk_weights_full, topk_ids_full = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k_full,
            use_grouped_topk=use_grouped_topk,
            renormalize=True,
            topk_group=kwargs.get("topk_group"),
            num_expert_group=kwargs.get("num_expert_group"),
            custom_routing_function=kwargs.get("custom_routing_function"),
            scoring_func=kwargs.get("scoring_func", "softmax"),
            routed_scaling_factor=kwargs.get("routed_scaling_factor", 1.0),
            e_score_correction_bias=kwargs.get("e_score_correction_bias"),
            num_experts=128,
            tid2eid=self.tid2eid,
            input_ids=input_ids,
        )

        # Step 2: Sort the 8 selected so zero experts (id >= 128) go to the end.
        is_zero = topk_ids_full >= 128  # [N, 8] bool
        sort_order = is_zero.int().argsort(dim=-1, stable=True)  # [N, 8]

        topk_ids_sorted = topk_ids_full.gather(1, sort_order)  # [N, 8]
        topk_weights_sorted = topk_weights_full.gather(1, sort_order)  # [N, 8]

        # Step 3: Take first K_refine=6.
        topk_ids_refine = topk_ids_sorted[:, :top_k].contiguous()  # [N, 6]
        topk_weights_refine = topk_weights_sorted[:, :top_k].contiguous()  # [N, 6]

        # Step 4: Clamp any remaining zero-expert ids to 0, weight to 0
        # (rare: >6 zero experts in top-8).
        zero_remaining = topk_ids_refine >= 128
        topk_ids_refine = torch.where(
            zero_remaining, torch.zeros_like(topk_ids_refine), topk_ids_refine
        )
        topk_weights_refine = torch.where(
            zero_remaining,
            torch.zeros_like(topk_weights_refine),
            topk_weights_refine,
        )

        topk_weights_refine = topk_weights_refine.to(x.dtype)

        # Step 5: Pass to GEMM pipeline (same as original apply, minus select_experts).
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        w13_weight_list = getattr(layer, "w13_weight_list", None)
        w2_weight_list = getattr(layer, "w2_weight_list", None)
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            w1 = w13_weight_list if isinstance(w13_weight_list, list) else [layer.w13_weight]
            w2 = w2_weight_list if isinstance(w2_weight_list, list) else [layer.w2_weight]
            w1_scale = [torch.tensor([], dtype=torch.int64)]
            w2_scale = [torch.tensor([], dtype=torch.int64)]
            w1_scale_bias = [torch.tensor([], dtype=torch.float32)]
            w2_scale_bias = [torch.tensor([], dtype=torch.float32)]
        else:
            w1 = w13_weight_list if isinstance(w13_weight_list, list) else layer.w13_weight
            w1_scale = None
            w2 = w2_weight_list if isinstance(w2_weight_list, list) else layer.w2_weight
            w2_scale = None
            w1_scale_bias = None
            w2_scale_bias = None

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights_refine,
                topk_ids=topk_ids_refine,
                w1=w1,
                w2=w2,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=kwargs.get("expert_map"),
                global_redundant_expert_num=kwargs.get("global_redundant_expert_num", 0),
                mc2_mask=kwargs.get("mc2_mask"),
                apply_router_weight_on_input=kwargs.get("apply_router_weight_on_input", False),
                log2phy=kwargs.get("log2phy"),
                pertoken_scale=kwargs.get("pertoken_scale"),
                activation=kwargs.get("activation", "silu"),
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
                lora_context=getattr(layer, "_ascend_moe_lora_context", None),
            )
        )
        return final_hidden_states


class Qwen3MoePlusPlusRefineSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """MoE block with refined zero-expert routing (graph-compatible).

    FusedMoE is configured with top_k=6 (for CANN active_num=N*6).
    RefineFusedMoEMethod.apply() does top-8 → sort → take 6 internally.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3MoeSparseMoeBlock, self).__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        zce_nums = list(getattr(config, "zce_nums", []) or [])
        num_zce = sum(zce_nums)
        total_experts = config.num_experts + num_zce
        self.num_zce = num_zce
        self.num_real_experts = config.num_experts

        self.top_k = config.num_experts_per_tok
        self.k_refine = self.top_k - 2
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = ReplicatedLinear(
            config.hidden_size,
            total_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", 0
        )
        if shared_expert_intermediate_size > 0:
            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.shared_expert_gate",
            )
            self.shared_expert = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert_gate = None
            self.shared_expert = None

        # FusedMoE with top_k=6 (K_refine) for CANN compatibility.
        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=None,
            num_experts=self.n_routed_experts,
            top_k=self.k_refine,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

        # Replace quant_method with RefineFusedMoEMethod.
        routed = getattr(self.experts, "routed_experts", None)
        if routed is not None:
            routed.quant_method = RefineFusedMoEMethod(
                self.experts.moe_config,
                tid2eid=getattr(self.experts, "tid2eid", None),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "Qwen3MoePlusPlusRefineSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Compute router_logits over all (real + zero) experts.
        router_logits, _ = self.gate(hidden_states)

        # Pass to self.experts — apply() does top-8 → sort → take 6 internally.
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            from vllm.distributed import tensor_model_parallel_all_gather

            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class Qwen3MoePlusPlusRefineDecoderLayer(Qwen3MoeDecoderLayer):
    """Decoder layer that instantiates the Refine MoE block for MoE layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoePlusPlusRefineSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        from vllm.model_executor.layers.layernorm import RMSNorm

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen3MoePlusPlusRefineModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen3MoePlusPlusRefineDecoderLayer,
        )


class Qwen3MoePlusPlusRefineForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3MoeForCausalLM, self).__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen3MoePlusPlusRefineModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts
