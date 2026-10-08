# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Qwen3.6 MLA decode-path optimizations (Tier 1).

Covers:
* D1 — ``Qwen3_6QProj.forward_q_and_gate`` returns the same query/gate as
  the original separate projection calls, and the per-forward stash in
  ``Qwen3_6MLAImplMixin`` serves the correct decode/prefill token ranges.
* D3 — the pure-decode fast path in ``Qwen3_6MLAImplMixin.forward`` builds
  the o_proj input directly from ``decode_output * sigmoid(gate)`` without
  a zero-initialized buffer, while padded/mixed batches keep the original
  zero-buffer path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.config import set_current_vllm_config

import vllm_ascend.models.qwen3_6_mla as qwen3_6_mla
from vllm_ascend.models.qwen3_6_mla import (
    Qwen3_6KVProjWithMQA,
    Qwen3_6MLAImplMixin,
    Qwen3_6QProj,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["none"]
    with set_current_vllm_config(mock_config):
        yield mock_config


class _StubParallelLinear(nn.Module):
    """Minimal ColumnParallelLinear/ReplicatedLinear stand-in with a weight."""

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.linear(x, self.weight), None


@pytest.fixture
def stub_parallel_linear(monkeypatch):
    monkeypatch.setattr(qwen3_6_mla, "ColumnParallelLinear", _StubParallelLinear)


def _make_q_proj(num_tokens: int, seed: int = 0) -> tuple[Qwen3_6QProj, torch.Tensor]:
    torch.manual_seed(seed)
    hidden_size, num_heads, head_dim = 16, 2, 8
    proj = Qwen3_6QProj(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        qk_head_dim=head_dim,
        rms_norm_eps=1e-6,
        prefix="test",
    )
    hidden = torch.randn(num_tokens, hidden_size)
    return proj, hidden


def test_forward_q_and_gate_matches_original_split(stub_parallel_linear):
    proj, hidden = _make_q_proj(num_tokens=7)

    q_flat, gate = proj.forward_q_and_gate(hidden)

    # Reference: the original separate-call semantics.
    q_and_gate = torch.nn.functional.linear(hidden, proj.qg_proj.weight)
    qg = q_and_gate.view(7, 2, 16)
    q_ref = proj.q_norm(qg[..., :8]).reshape(7, 16)
    gate_ref = qg[..., 8:].reshape(7, 16)

    assert q_flat.shape == (7, 16)
    assert gate.shape == (7, 16)
    assert torch.equal(q_flat, q_ref)
    assert torch.equal(gate, gate_ref)


def test_forward_and_get_gate_delegate_to_fused_call(stub_parallel_linear):
    proj, hidden = _make_q_proj(num_tokens=5)

    q_flat, gate = proj.forward_q_and_gate(hidden)

    assert torch.equal(proj.forward(hidden)[0], q_flat)
    assert torch.equal(proj.get_gate(hidden), gate)


def _make_bare_impl(num_heads: int = 2, qk_head_dim: int = 8, rope_dim: int = 4, lora_rank: int = 6):
    impl = object.__new__(Qwen3_6MLAImplMixin)
    impl.num_heads = num_heads
    impl.num_kv_heads = 1
    impl.qk_head_dim = qk_head_dim
    impl.qk_rope_head_dim = rope_dim
    impl.qk_nope_head_dim = qk_head_dim - rope_dim
    impl.v_head_dim = 8
    impl.kv_lora_rank = lora_rank
    impl.W_UK_T = torch.randn(num_heads, qk_head_dim - rope_dim, lora_rank)
    impl._precomputed_q_flat = None
    impl._precomputed_q_num_decode_tokens = 0
    return impl


def test_apply_rope_neox_cpu_fallback_matches_reference():
    impl = _make_bare_impl(rope_dim=4)
    torch.manual_seed(1)
    for x_shape in [(7, 2, 4), (7, 2, 1, 4)]:
        x = torch.randn(*x_shape)
        cos = torch.randn(7, 1, 1, 4)
        sin = torch.randn(7, 1, 1, 4)

        out = impl._apply_rope_neox(x, cos, sin)

        x4 = x if x.dim() == 4 else x.unsqueeze(2)
        x1 = x4[..., :2]
        x2 = x4[..., 2:]
        ref = x4 * cos + torch.cat((-x2, x1), dim=-1) * sin
        ref = ref.squeeze(2) if x.dim() == 3 else ref

        assert out.shape == x.shape
        assert torch.equal(out, ref)


@pytest.fixture
def stub_replicated_linear(monkeypatch):
    monkeypatch.setattr(qwen3_6_mla, "ReplicatedLinear", _StubParallelLinear)


def _make_kv_proj():
    return Qwen3_6KVProjWithMQA(
        hidden_size=16,
        num_kv_heads=2,
        head_dim=8,
        rope_dim=2,
        kv_lora_rank=12,
        rms_norm_eps=1e-6,
    )


def test_kv_proj_merged_gemm_matches_separate(stub_replicated_linear):
    torch.manual_seed(2)
    module = _make_kv_proj()
    hidden = torch.randn(9, 16)

    out_separate = module(hidden)[0]
    assert module._merged_kv_weight is None

    module.merge_kv_weights()
    assert module._merged_kv_weight is not None
    out_merged = module(hidden)[0]

    assert out_merged.shape == (9, 14)
    assert torch.equal(out_merged, out_separate)


def test_merge_kv_weights_skips_quantized(stub_replicated_linear):
    module = _make_kv_proj()
    module.k_proj.quant_method = object()  # not UnquantizedLinearMethod

    module.merge_kv_weights()

    assert module._merged_kv_weight is None


def test_q_proj_and_k_up_proj_uses_stashed_decode_range():
    impl = _make_bare_impl()
    num_decode, num_actual = 3, 7
    q_flat = torch.randn(num_actual, impl.num_heads * impl.qk_head_dim)
    impl._precomputed_q_flat = q_flat
    impl._precomputed_q_num_decode_tokens = num_decode

    ql_nope, q_pe = impl._q_proj_and_k_up_proj(torch.empty(0))

    q = q_flat[:num_decode].view(-1, impl.num_heads, impl.qk_head_dim)
    q_pe_ref = q[..., : impl.qk_rope_head_dim]
    q_nope_ref = q[..., impl.qk_rope_head_dim :]
    # Reference: broadcast (B, N, 1, P) x (N, P, L) -> (B, N, 1, L)
    ql_nope_ref = torch.matmul(q_nope_ref.unsqueeze(2), impl.W_UK_T)

    assert ql_nope.shape == (num_decode, impl.num_heads, 1, impl.kv_lora_rank)
    assert torch.equal(ql_nope, ql_nope_ref)
    assert torch.equal(q_pe, q_pe_ref)


def test_q_proj_and_k_up_proj_falls_back_without_stash():
    impl = _make_bare_impl()
    x = torch.randn(4, 16)
    impl.q_proj = MagicMock()
    impl.q_proj.return_value = (torch.randn(4, impl.num_heads * impl.qk_head_dim), None)

    ql_nope, q_pe = impl._q_proj_and_k_up_proj(x)

    impl.q_proj.assert_called_once_with(x)
    assert ql_nope.shape == (4, impl.num_heads, 1, impl.kv_lora_rank)


def test_mla_preprocess_prefill_uses_stashed_prefill_range():
    impl = _make_bare_impl()
    num_decode, num_actual = 3, 7
    num_prefill = num_actual - num_decode
    heads, qk, rope = impl.num_heads, impl.qk_head_dim, impl.qk_rope_head_dim

    q_flat = torch.randn(num_actual, heads * qk)
    impl._precomputed_q_flat = q_flat
    impl._precomputed_q_num_decode_tokens = num_decode

    cos = torch.randn(1, 1, 1, rope)
    sin = torch.randn(1, 1, 1, rope)
    k_pe_ref = torch.randn(num_prefill, impl.num_kv_heads, rope)
    kv_c_ref = torch.randn(num_prefill, impl.kv_lora_rank)
    impl.exec_kv_prefill = MagicMock(return_value=(k_pe_ref, kv_c_ref))

    kv_b_out = torch.randn(num_prefill, heads * (impl.qk_nope_head_dim + impl.v_head_dim))
    impl.kv_b_proj = MagicMock(return_value=(kv_b_out, None))

    attn_metadata = SimpleNamespace(
        num_decode_tokens=num_decode,
        num_actual_tokens=num_actual,
        prefill=SimpleNamespace(cos=cos, sin=sin),
        slot_mapping=torch.arange(num_actual),
    )
    hidden = torch.randn(num_actual, 16)
    kv_no_split = torch.randn(num_actual, impl.kv_lora_rank + rope)

    res = impl.mla_preprocess_prefill(hidden, kv_no_split, (None,), attn_metadata)

    prefill_q = q_flat[num_decode:].view(-1, heads, qk)
    assert torch.equal(res.q_nope, prefill_q[..., rope:])
    assert torch.equal(res.q_pe, impl.rope_single(prefill_q[..., :rope], cos, sin))
    # exec_kv_prefill must see the prefill token range only.
    assert impl.exec_kv_prefill.call_args[0][0].shape[0] == num_prefill
    # The stash must not be consumed by the (mocked) q_proj projection.
    assert res.k_nope.shape == (num_prefill, heads, impl.qk_nope_head_dim)
    assert res.value.shape == (num_prefill, heads, impl.v_head_dim)


class _StubImpl(Qwen3_6MLAImplMixin):
    """Bare impl with just enough surface for ``forward`` to run."""

    def __init__(self, num_heads: int = 2, v_head_dim: int = 8):
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self._precomputed_q_flat = None
        self._precomputed_q_num_decode_tokens = 0

    def get_num_actual_tokens(self, attn_metadata):
        return attn_metadata.num_actual_tokens


def _run_forward(num_actual: int, num_decode: int, padded_tokens: int, has_prefill: bool):
    torch.manual_seed(0)
    impl = _StubImpl()
    hidden = torch.randn(num_actual, 16)
    gate = torch.randn(num_actual, impl.num_heads * impl.v_head_dim)
    decode_out = torch.randn(num_decode, impl.num_heads * impl.v_head_dim)

    impl.q_proj = MagicMock()
    impl.q_proj.forward_q_and_gate = MagicMock(return_value=(torch.empty(0), gate))
    decode_res = SimpleNamespace(
        ql_nope=None, q_pe=None, k_nope=None, k_pe=None, dequant_scale_q_nope=None
    )
    prefill_res = None
    if has_prefill:
        prefill_res = SimpleNamespace(
            q_nope=None, q_pe=None, k_nope=None, k_pe=None, value=torch.randn(0, 8)
        )
    impl._mla_preprocess = MagicMock(return_value=(decode_res, prefill_res))
    impl._forward_decode = MagicMock(return_value=decode_out)
    impl._forward_prefill = MagicMock(
        return_value=torch.randn(num_actual - num_decode, impl.num_heads * impl.v_head_dim)
    )
    impl.o_proj = MagicMock(side_effect=lambda inp, is_prefill: (torch.full((num_actual, 3), 7.0),))

    attn_metadata = SimpleNamespace(
        num_decodes=1 if num_decode else 0,
        num_prefills=1 if has_prefill else 0,
        num_decode_tokens=num_decode,
        num_actual_tokens=num_actual,
    )
    output = torch.empty(num_actual, 3)
    kv_cache = (
        torch.zeros(1, 128, 1, 16),
        torch.zeros(1, 128, 1, 4),
    )

    with (
        patch(
            "vllm_ascend.ascend_forward_context._EXTRA_CTX",
            SimpleNamespace(num_tokens=padded_tokens),
        ),
        patch("vllm_ascend.attention.utils.maybe_save_kv_layer_to_connector"),
        patch.object(
            torch.ops.vllm,
            "maybe_all_gather_and_maybe_unpad",
            lambda x, label, is_ep_comm=False: x,
            create=True,
        ),
    ):
        impl.forward("layer0", hidden, kv_cache, attn_metadata, output=output)

    o_proj_input = impl.o_proj.call_args[0][0]
    return gate, decode_out, o_proj_input


def test_forward_pure_decode_builds_input_directly():
    num_tokens = 5
    gate, decode_out, o_proj_input = _run_forward(
        num_actual=num_tokens, num_decode=num_tokens, padded_tokens=num_tokens, has_prefill=False
    )

    expected = decode_out * torch.sigmoid(gate)
    assert o_proj_input.shape == (num_tokens, 16)
    assert torch.equal(o_proj_input, expected)


def test_forward_padded_decode_keeps_zero_buffer():
    num_actual, padded = 5, 8
    gate, decode_out, o_proj_input = _run_forward(
        num_actual=num_actual, num_decode=num_actual, padded_tokens=padded, has_prefill=False
    )

    expected = decode_out * torch.sigmoid(gate)
    assert o_proj_input.shape == (padded, 16)
    assert torch.equal(o_proj_input[:num_actual], expected)
    assert torch.equal(o_proj_input[num_actual:], torch.zeros(padded - num_actual, 16))


def test_forward_mixed_batch_fills_both_ranges():
    num_actual, num_decode = 7, 3
    gate, decode_out, o_proj_input = _run_forward(
        num_actual=num_actual, num_decode=num_decode, padded_tokens=num_actual, has_prefill=True
    )

    assert o_proj_input.shape == (num_actual, 16)
    expected_decode = decode_out * torch.sigmoid(gate[:num_decode])
    assert torch.equal(o_proj_input[:num_decode], expected_decode)
    # Prefill rows are nonzero and distinct from the zero buffer.
    assert o_proj_input[num_decode:].abs().sum() > 0
