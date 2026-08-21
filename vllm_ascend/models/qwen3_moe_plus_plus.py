# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from vllm/model_executor/models/qwen3_moe.py.
# Qwen3 MoE++ (ZEDA) variant: adds Zero/Copy/Constant experts (ZCE) on top of
# the stock Qwen3MoE. Zero experts carry no weights and output zero; routing
# tokens to them attenuates real-expert contributions via top-k renormalization
# (dynamic MoE activation). The gate outputs (num_experts + sum(zce_nums))
# logits; zero experts occupy the trailing indices and are post-processed by
# vllm-ascend's zero_experts_compute (threshold = num_logical_experts).

from typing import Any

import torch
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


class Qwen3MoePlusPlusSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """MoE block with an external gate over (real + zero) experts.

    The gate is NOT passed into FusedMoE (so is_internal_router=False and the
    default softmax router is used, no e_score_correction_bias required). The
    forward computes the (num_experts + num_zce)-dim router_logits and hands it
    to the runner; vllm-ascend's select_experts routes over the full gate dim
    and zero_experts_compute (threshold=num_logical_experts) remaps the zero
    experts to expert 0 with weight 0.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Skip Qwen3MoeSparseMoeBlock.__init__ (it would build a stock 128-dim
        # gate + a FusedMoE we would immediately discard, double-counting the
        # Ascend MoE layer id). Initialize nn.Module and replicate the setup.
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

        # ZCE config.
        zce_nums = list(getattr(config, "zce_nums", []) or [])
        zce_types = list(getattr(config, "zce_types", []) or [])
        num_zce = sum(zce_nums)
        zce_type = zce_types[0] if zce_types else "zero"
        total_experts = config.num_experts + num_zce
        self.num_zce = num_zce
        self.zce_type = zce_type
        self.use_zce_mask = bool(getattr(config, "use_zce_mask", False))

        # External gate over (real + zero) experts. Not passed to FusedMoE so
        # the runner uses external router_logits (is_internal_router=False).
        self.gate = ReplicatedLinear(
            config.hidden_size,
            total_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # Shared expert (Qwen3-30B-A3B has none; kept for generality).
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

        # Only the real experts have weights; zero experts are handled by
        # zero_experts_compute. Pass gate=None so the factory does not create a
        # ZeroExpertRouter (which would require e_score_correction_bias).
        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=None,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # The FusedMoE factory does not propagate these to routed_experts, but
        # AscendMoERunner.apply reads them via getattr(layer, ...). Set them so
        # the zero-expert post-processing path activates.
        routed = getattr(self.experts, "routed_experts", None)
        if routed is not None:
            routed.zero_expert_num = num_zce
            routed.zero_expert_type = zce_type


class Qwen3MoePlusPlusDecoderLayer(Qwen3MoeDecoderLayer):
    """Decoder layer that instantiates the PlusPlus MoE block for MoE layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Skip Qwen3MoeDecoderLayer.__init__ and replicate, swapping in the
        # PlusPlus MoE block (avoids building a stock MoE block first).
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
            self.mlp = Qwen3MoePlusPlusSparseMoeBlock(
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


class Qwen3MoePlusPlusModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen3MoePlusPlusDecoderLayer,
        )


class Qwen3MoePlusPlusForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Skip Qwen3MoeForCausalLM.__init__ (it would construct a stock
        # Qwen3MoeModel and double-allocate expert weights). Initialize the
        # nn.Module + interface mixins in the MRO above Qwen3MoeForCausalLM,
        # then replicate the body with the PlusPlus model class.
        super(Qwen3MoeForCausalLM, self).__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen3MoePlusPlusModel(
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

        # MoE hyperparameters (mirrors Qwen3MoeForCausalLM).
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
