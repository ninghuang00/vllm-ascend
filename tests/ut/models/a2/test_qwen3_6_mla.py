# SPDX-License-Identifier: Apache-2.0
"""NPU unit tests for the Qwen3.6 MLA decode cache-write path (D2).

Verifies that ``Qwen3_6MLAImplMixin.exec_kv_decode`` — which scatters the
latent / rope tensors into the paged caches via the fused
``torch_npu.npu_scatter_pa_kv_cache`` op — produces bit-identical cache
contents compared with the original per-layer floor_div/floor_mod +
index_put reference.
"""

import pytest
import torch
from torch import nn
from unittest.mock import MagicMock
from vllm.config import set_current_vllm_config

import vllm_ascend.models.qwen3_6_mla as qwen3_6_mla
from vllm_ascend.models.qwen3_6_mla import (
    Qwen3_6KVProjWithMQA,
    Qwen3_6MLAImplMixin,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["none"]
    with set_current_vllm_config(mock_config):
        yield mock_config


def _make_impl(kv_lora_rank: int, rope_dim: int) -> Qwen3_6MLAImplMixin:
    impl = object.__new__(Qwen3_6MLAImplMixin)
    impl.num_kv_heads = 1
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = rope_dim
    impl._precomputed_q_flat = None
    impl._precomputed_q_num_decode_tokens = 0
    return impl


@pytest.mark.parametrize(
    ("num_tokens", "kv_lora_rank", "rope_dim"),
    [
        (1, 512, 64),
        (7, 512, 64),
        (64, 512, 64),
        (64, 128, 16),
    ],
)
def test_exec_kv_decode_scatter_matches_index_put(num_tokens, kv_lora_rank, rope_dim):
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)
    device = "npu"
    num_blocks, block_size = 8, 128

    kv_no_split = torch.randn(num_tokens, kv_lora_rank + rope_dim, dtype=torch.bfloat16, device=device)
    # cos/sin shaped like the decode metadata rotary tables: [B, 1, 1, rope].
    cos = torch.randn(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device)
    slots = torch.randperm(num_blocks * block_size, device=device)[:num_tokens].to(torch.int64)

    fused_caches = (
        torch.zeros(num_blocks, block_size, 1, kv_lora_rank, dtype=torch.bfloat16, device=device),
        torch.zeros(num_blocks, block_size, 1, rope_dim, dtype=torch.bfloat16, device=device),
    )
    ref_caches = (
        torch.zeros(num_blocks, block_size, 1, kv_lora_rank, dtype=torch.bfloat16, device=device),
        torch.zeros(num_blocks, block_size, 1, rope_dim, dtype=torch.bfloat16, device=device),
    )

    impl = _make_impl(kv_lora_rank, rope_dim)

    ret_k_pe_cache, ret_kv_c_cache = impl.exec_kv_decode(kv_no_split, cos, sin, fused_caches, slots)
    assert ret_k_pe_cache is fused_caches[1]
    assert ret_kv_c_cache is fused_caches[0]

    # Reference: the original floor_div/floor_mod + index_put path.
    kv = kv_no_split.view(num_tokens, 1, 1, kv_lora_rank + rope_dim)
    kv_c, k_pe = kv.split([kv_lora_rank, rope_dim], dim=-1)
    k_pe = impl._apply_rope_neox(k_pe, cos, sin)
    block_idx = (slots // block_size).long()
    block_off = (slots % block_size).long()
    ref_caches[0][block_idx, block_off] = kv_c.squeeze(2)
    ref_caches[1][block_idx, block_off] = k_pe.squeeze(2)

    torch.npu.synchronize()
    assert torch.equal(fused_caches[0], ref_caches[0])
    assert torch.equal(fused_caches[1], ref_caches[1])


def test_exec_kv_decode_nz_fallback_matches_index_put():
    torch.manual_seed(0)
    device = "npu"
    num_tokens, kv_lora_rank, rope_dim = 16, 512, 64
    num_blocks, block_size = 4, 128

    kv_no_split = torch.randn(num_tokens, kv_lora_rank + rope_dim, dtype=torch.bfloat16, device=device)
    cos = torch.randn(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device)
    slots = torch.randperm(num_blocks * block_size, device=device)[:num_tokens].to(torch.int64)

    caches = (
        torch.zeros(num_blocks, block_size, 1, kv_lora_rank, dtype=torch.bfloat16, device=device),
        torch.zeros(num_blocks, block_size, 1, rope_dim, dtype=torch.bfloat16, device=device),
    )
    ref = tuple(c.clone() for c in caches)

    impl = _make_impl(kv_lora_rank, rope_dim)
    impl.enable_kv_nz = True

    impl.exec_kv_decode(kv_no_split, cos, sin, caches, slots)

    kv = kv_no_split.view(num_tokens, 1, 1, kv_lora_rank + rope_dim)
    kv_c, k_pe = kv.split([kv_lora_rank, rope_dim], dim=-1)
    k_pe = impl._apply_rope_neox(k_pe, cos, sin)
    block_idx = (slots // block_size).long()
    block_off = (slots % block_size).long()
    ref[0][block_idx, block_off] = kv_c.squeeze(2)
    ref[1][block_idx, block_off] = k_pe.squeeze(2)

    torch.npu.synchronize()
    assert torch.equal(caches[0], ref[0])
    assert torch.equal(caches[1], ref[1])


# ── D5: fused NeoX RoPE via torch_npu.npu_rotary_mul ───────────────────


def _rope_fp32_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x.float() * cos.float() + rot.float() * sin.float()


@pytest.mark.parametrize(("num_tokens", "num_heads"), [(1, 8), (7, 8), (64, 8), (64, 1)])
def test_apply_rope_neox_fused_matches_fp32_reference(num_tokens, num_heads):
    """The fused op accumulates in fp32 and rounds once, so its bf16 output
    must be bit-identical to the fp32 reference rounded to bf16."""
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)
    device = "npu"
    rope_dim = 64
    impl = _make_impl(512, rope_dim)

    x = torch.randn(num_tokens, num_heads, 1, rope_dim, dtype=torch.bfloat16, device=device) * 2
    cos = torch.rand(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device) * 2 - 1
    sin = torch.rand(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device) * 2 - 1

    out = impl._apply_rope_neox(x, cos, sin)

    expected = _rope_fp32_reference(x, cos, sin).to(torch.bfloat16)
    assert torch.equal(out, expected)


def test_apply_rope_neox_3d_input_fused():
    torch.manual_seed(0)
    device = "npu"
    rope_dim = 64
    impl = _make_impl(512, rope_dim)
    x = torch.randn(64, 8, rope_dim, dtype=torch.bfloat16, device=device)
    cos = torch.rand(64, 1, 1, rope_dim, dtype=torch.bfloat16, device=device) * 2 - 1
    sin = torch.rand(64, 1, 1, rope_dim, dtype=torch.bfloat16, device=device) * 2 - 1

    out = impl._apply_rope_neox(x, cos, sin)

    expected = _rope_fp32_reference(x.unsqueeze(2), cos, sin).to(torch.bfloat16).squeeze(2)
    assert out.shape == x.shape
    assert torch.equal(out, expected)


def test_apply_rope_neox_large_batch_falls_back_to_python_chain():
    """Token batches beyond the op's B < 1000 constraint (prefill-sized)
    must take the original PyTorch chain, bit-identical to it."""
    torch.manual_seed(0)
    device = "npu"
    rope_dim = 64
    impl = _make_impl(512, rope_dim)
    num_tokens = 1200
    x = torch.randn(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device)
    cos = torch.rand(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device) * 2 - 1
    sin = torch.rand(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16, device=device) * 2 - 1

    out = impl._apply_rope_neox(x, cos, sin)

    x1 = x[..., : rope_dim // 2]
    x2 = x[..., rope_dim // 2 :]
    ref = x * cos + torch.cat((-x2, x1), dim=-1) * sin
    assert torch.equal(out, ref)


# ── D6: merged k/v GEMM + broadcast matmul ────────────────────────────


class _StubReplicatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(
            (torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.05)
        )
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.linear(x, self.weight), None


def test_kv_proj_merged_gemm_matches_separate(monkeypatch):
    monkeypatch.setattr(qwen3_6_mla, "ReplicatedLinear", _StubReplicatedLinear)
    torch.manual_seed(3)
    device = "npu"
    module = Qwen3_6KVProjWithMQA(
        hidden_size=2048,
        num_kv_heads=2,
        head_dim=256,
        rope_dim=64,
        kv_lora_rank=512,
        rms_norm_eps=1e-6,
    ).npu()
    hidden = torch.randn(64, 2048, dtype=torch.bfloat16, device=device)

    out_separate = module(hidden)[0]
    module.merge_kv_weights()
    out_merged = module(hidden)[0]

    assert out_merged.shape == (64, 576)
    assert torch.equal(out_merged, out_separate)


def test_q_proj_and_k_up_proj_broadcast_matmul_matches_bmm():
    torch.manual_seed(4)
    device = "npu"
    num_heads, qk_head_dim, rope_dim, lora_rank = 8, 256, 64, 512
    impl = object.__new__(Qwen3_6MLAImplMixin)
    impl.num_heads = num_heads
    impl.qk_head_dim = qk_head_dim
    impl.qk_rope_head_dim = rope_dim
    impl.kv_lora_rank = lora_rank
    impl.W_UK_T = torch.randn(num_heads, qk_head_dim - rope_dim, lora_rank, dtype=torch.bfloat16, device=device) * 0.02
    impl._precomputed_q_flat = None
    impl._precomputed_q_num_decode_tokens = 0
    impl.q_proj = MagicMock()
    impl.q_proj.return_value = (
        torch.randn(64, num_heads * qk_head_dim, dtype=torch.bfloat16, device=device),
        None,
    )

    ql_nope, q_pe = impl._q_proj_and_k_up_proj(torch.empty(0, 2048, device=device))

    q = impl.q_proj.return_value[0].view(-1, num_heads, qk_head_dim)
    q_nope = q[..., rope_dim:]
    ref = (
        torch.bmm(q_nope.transpose(0, 1), impl.W_UK_T)
        .transpose(0, 1)
        .contiguous()
        .view(64, num_heads, 1, lora_rank)
    )

    assert ql_nope.shape == (64, num_heads, 1, lora_rank)
    assert ql_nope.is_contiguous()
    assert torch.equal(ql_nope, ref)
    assert torch.equal(q_pe, q[..., :rope_dim])
