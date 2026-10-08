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

from vllm_ascend.models.qwen3_6_mla import Qwen3_6MLAImplMixin


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
