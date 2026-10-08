# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Fused MLA preprocessing kernels for the Qwen3.6 compressed-KV model.

These two kernels replace the ~27 per-layer fragment ops of the MLA path
(see docs/7.2_fused_kernel_implementation_plan.md):

* ``fused_knorm_rearrange`` (K1) — per-head Gemma RMSNorm on the K half of
  ``kv_proj`` output plus the [k_rot | k_pass | v] rearrangement that feeds
  ``kv_a_proj``.  Replaces 1 norm + 2 slices + 1 concat.

* ``fused_mla_rope_scatter_split`` (K2) — Q-side Gemma RMSNorm + NeoX RoPE +
  q_pe/q_nope split + gate copy, and KV-side k_pe NeoX RoPE + kv_c/k_pe split
  + paged-cache scatter.  Replaces the remaining norm/slice/neg/cat/mul/add
  and IndexPutV2 fragments.

Numerics match the unfused eager path on 910B3 (aclnnGemmaRmsNorm /
bf16 elementwise rope):
* Gemma norm: ``x * (1/sqrt(mean+eps)) * (1 + w)`` fully in fp32 (the +1 in
  fp32, like aclnnGemmaRmsNorm), single round to bf16 on store.
* NeoX rope: per-op bf16 rounding ``bf16(x*cos) + bf16(rot(x)*sin)`` (matches
  the eager ``x * cos + rotate_half(x) * sin`` on bf16 tensors).
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import (
    extract_slice,
    get_vectorcore_num,
    init_device_properties_triton,
    insert_slice,
)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "front_core_num",
        "num_tokens_each_front_core",
        "num_tokens_each_tail_core",
    ]
)
def fused_knorm_rearrange_kernel(
    kv_in_ptr: torch.Tensor,
    k_weight_ptr: torch.Tensor,
    kv_out_ptr: torch.Tensor,
    num_tokens,
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM
    K_SIZE: tl.constexpr = KV_HEADS * HEAD_DIM
    KV_ROW: tl.constexpr = K_SIZE * 2

    block_idx = tl.program_id(0)

    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core

    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    # (1 + w) computed in fp32, matching aclnnGemmaRmsNorm's internal gain
    k_gamma = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32) + 1.0

    h_idx = tl.arange(0, KV_HEADS)
    d_idx = tl.arange(0, HEAD_DIM)
    r_idx = tl.arange(0, ROPE_DIM)
    n_idx = tl.arange(0, NOPE_DIM)

    for index in range(loop_num):
        row = block_offset + index

        k = tl.load(kv_in_ptr + row * KV_ROW + h_idx[:, None] * HEAD_DIM + d_idx[None, :]).to(tl.float32)

        squares = k * k
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = tl.div_rn(1.0, tl.sqrt_rn(variances + EPS))
        k_norm = k * reciprocal_std[:, None] * k_gamma[None, :]

        # rearrange: [k_rot(128) | k_pass(384) | v(512)]
        k_rot = extract_slice(k_norm, (0, 0), (KV_HEADS, ROPE_DIM), (1, 1)).to(tl.bfloat16)
        tl.store(kv_out_ptr + row * KV_ROW + h_idx[:, None] * ROPE_DIM + r_idx[None, :], k_rot)

        k_pass = extract_slice(k_norm, (0, ROPE_DIM), (KV_HEADS, NOPE_DIM), (1, 1)).to(tl.bfloat16)
        tl.store(
            kv_out_ptr + row * KV_ROW + KV_HEADS * ROPE_DIM + h_idx[:, None] * NOPE_DIM + n_idx[None, :], k_pass
        )

        v = tl.load(kv_in_ptr + row * KV_ROW + K_SIZE + h_idx[:, None] * HEAD_DIM + d_idx[None, :])
        tl.store(kv_out_ptr + row * KV_ROW + K_SIZE + h_idx[:, None] * HEAD_DIM + d_idx[None, :], v)


def triton_fused_knorm_rearrange(
    kv: torch.Tensor,
    k_weight: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
) -> tuple[torch.Tensor]:
    """Fuse per-head Gemma RMSNorm on K + [k_rot|k_pass|v] rearrangement.

    Args:
        kv: ``kv_proj`` output ``[num_tokens, 2 * num_kv_heads * head_dim]``
            laid out as [k(num_kv_heads*head_dim) | v(num_kv_heads*head_dim)].
        k_weight: Gemma norm weight ``[head_dim]`` (raw; the +1 is applied
            in fp32 inside the kernel, matching aclnnGemmaRmsNorm).
    Returns:
        Rearranged ``[num_tokens, 2 * num_kv_heads * head_dim]`` tensor
        [k_rot | k_pass | v] feeding ``kv_a_proj``.
    """
    assert kv.dtype == torch.bfloat16, "fused_knorm_rearrange only supports bf16"
    num_tokens = kv.shape[0]
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return (kv_out,)

    # idempotent: worker.py calls it once at serve time; standalone scripts
    # need it before querying the vector core count.
    init_device_properties_triton()
    core_num = get_vectorcore_num()

    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num
    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num
    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num
    num_tokens_each_tail_core = num_tokens // core_num

    total_core = front_core_num + tail_core_num
    block_dim = core_num
    if total_core < core_num:
        block_dim = total_core

    fused_knorm_rearrange_kernel[(block_dim,)](
        kv,
        k_weight,
        kv_out,
        num_tokens,
        front_core_num,
        num_tokens_each_front_core,
        num_tokens_each_tail_core,
        num_kv_heads,
        head_dim,
        rope_dim,
        eps,
    )
    return (kv_out,)


def triton_fused_knorm_rearrange_fake(
    kv: torch.Tensor,
    k_weight: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
) -> tuple[torch.Tensor]:
    return (torch.empty_like(kv),)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "front_core_num",
        "num_tokens_each_front_core",
        "num_tokens_each_tail_core",
        "block_size",
    ]
)
def fused_mla_rope_scatter_split_kernel(
    qg_ptr: torch.Tensor,
    q_weight_ptr: torch.Tensor,
    kv_ptr: torch.Tensor,
    cos_ptr: torch.Tensor,
    sin_ptr: torch.Tensor,
    slots_ptr: torch.Tensor,
    cache_c_ptr: torch.Tensor,
    cache_pe_ptr: torch.Tensor,
    gate_ptr: torch.Tensor,
    q_pe_ptr: torch.Tensor,
    q_nope_ptr: torch.Tensor,
    kv_c_ret_ptr: torch.Tensor,
    k_pe_ret_ptr: torch.Tensor,
    num_tokens,
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    block_size,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    KV_RANK: tl.constexpr,
    EPS: tl.constexpr,
):
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2
    QG_ROW: tl.constexpr = H * HEAD_DIM * 2
    KV_ROW: tl.constexpr = KV_RANK + ROPE_DIM

    block_idx = tl.program_id(0)

    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core

    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    # (1 + w) computed in fp32, matching aclnnGemmaRmsNorm's internal gain
    q_gamma = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32) + 1.0

    h_idx = tl.arange(0, H)
    qg_d_idx = tl.arange(0, HEAD_DIM * 2)
    r_idx = tl.arange(0, ROPE_DIM)
    n_idx = tl.arange(0, NOPE_DIM)
    g_idx = tl.arange(0, HEAD_DIM)
    c_idx = tl.arange(0, KV_RANK)

    for index in range(loop_num):
        row = block_offset + index

        cos_t = tl.load(cos_ptr + row * ROPE_DIM + r_idx).to(tl.float32)
        sin_t = tl.load(sin_ptr + row * ROPE_DIM + r_idx).to(tl.float32)
        cos_q = tl.broadcast_to(cos_t.reshape(1, ROPE_DIM), (H, ROPE_DIM))
        sin_q = tl.broadcast_to(sin_t.reshape(1, ROPE_DIM), (H, ROPE_DIM))

        # ── Q side: norm + NeoX rope + split + gate ──
        qg_t = tl.load(qg_ptr + row * QG_ROW + h_idx[:, None] * HEAD_DIM * 2 + qg_d_idx[None, :]).to(tl.float32)
        q = extract_slice(qg_t, (0, 0), (H, HEAD_DIM), (1, 1))

        squares = q * q
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = tl.div_rn(1.0, tl.sqrt_rn(variances + EPS))
        # round to bf16 once (norm output), then stay in fp32 for the rope so
        # the per-op bf16 rounding below reproduces the eager bf16 ops exactly
        qn = (q * reciprocal_std[:, None] * q_gamma[None, :]).to(tl.bfloat16).to(tl.float32)

        q_pe = extract_slice(qn, (0, 0), (H, ROPE_DIM), (1, 1))
        x1 = extract_slice(q_pe, (0, 0), (H, HALF_ROPE), (1, 1))
        x2 = extract_slice(q_pe, (0, HALF_ROPE), (H, HALF_ROPE), (1, 1))
        rot = tl.zeros((H, ROPE_DIM), dtype=tl.float32)
        rot = insert_slice(rot, -x2, (0, 0), (H, HALF_ROPE), (1, 1))
        rot = insert_slice(rot, x1, (0, HALF_ROPE), (H, HALF_ROPE), (1, 1))
        a = (q_pe * cos_q).to(tl.bfloat16)
        b = (rot * sin_q).to(tl.bfloat16)
        q_pe_rope = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        tl.store(q_pe_ptr + row * (H * ROPE_DIM) + h_idx[:, None] * ROPE_DIM + r_idx[None, :], q_pe_rope)

        q_nope = extract_slice(qn, (0, ROPE_DIM), (H, NOPE_DIM), (1, 1)).to(tl.bfloat16)
        tl.store(q_nope_ptr + row * (H * NOPE_DIM) + h_idx[:, None] * NOPE_DIM + n_idx[None, :], q_nope)

        gate = extract_slice(qg_t, (0, HEAD_DIM), (H, HEAD_DIM), (1, 1)).to(tl.bfloat16)
        tl.store(gate_ptr + row * (H * HEAD_DIM) + h_idx[:, None] * HEAD_DIM + g_idx[None, :], gate)

        # ── KV side: k_pe rope + kv_c/k_pe split ──
        kc = tl.load(kv_ptr + row * KV_ROW + c_idx)
        kp = tl.load(kv_ptr + row * KV_ROW + KV_RANK + r_idx).to(tl.float32).reshape(1, ROPE_DIM)
        kp1 = extract_slice(kp, (0, 0), (1, HALF_ROPE), (1, 1))
        kp2 = extract_slice(kp, (0, HALF_ROPE), (1, HALF_ROPE), (1, 1))
        krot = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
        krot = insert_slice(krot, -kp2, (0, 0), (1, HALF_ROPE), (1, 1))
        krot = insert_slice(krot, kp1, (0, HALF_ROPE), (1, HALF_ROPE), (1, 1))
        ka = (kp * cos_t.reshape(1, ROPE_DIM)).to(tl.bfloat16)
        kb = (krot * sin_t.reshape(1, ROPE_DIM)).to(tl.bfloat16)
        kpe = (ka.to(tl.float32) + kb.to(tl.float32)).to(tl.bfloat16).reshape(ROPE_DIM)

        tl.store(kv_c_ret_ptr + row * KV_RANK + c_idx, kc)
        tl.store(k_pe_ret_ptr + row * ROPE_DIM + r_idx, kpe)

        # ── paged-cache scatter (skip PAD_SLOT_ID < 0) ──
        # cache layout [num_blocks, block_size, 1, dim] is contiguous, so the
        # flat offset of slot s is simply s * dim.
        slot = tl.load(slots_ptr + row)
        c_mask = (slot >= 0) & (c_idx >= 0)
        r_mask = (slot >= 0) & (r_idx >= 0)
        tl.store(cache_c_ptr + slot * KV_RANK + c_idx, kc, mask=c_mask)
        tl.store(cache_pe_ptr + slot * ROPE_DIM + r_idx, kpe, mask=r_mask)


def triton_fused_mla_rope_scatter_split(
    qg: torch.Tensor,
    q_weight: torch.Tensor,
    kv576: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    cache_c: torch.Tensor,
    cache_pe: torch.Tensor,
    gate_out: torch.Tensor,
    num_heads: int,
    kv_lora_rank: int,
    rope_dim: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q norm/rope/split/gate + KV rope/split + paged-cache scatter.

    Args:
        qg: ``qg_proj`` output ``[num_tokens, num_heads * (head_dim*2)]`` with
            per-head interleaved [q(head_dim) | gate(head_dim)] blocks.
        q_weight: Q Gemma norm weight ``[head_dim]`` (raw; the +1 is applied
            in fp32 inside the kernel, matching aclnnGemmaRmsNorm).
        kv576: ``kv_a_proj`` output ``[num_tokens, kv_lora_rank + rope_dim]``.
        cos/sin: rotary tables ``[num_tokens, 1, 1, rope_dim]`` (contiguous).
        slots: slot mapping ``[num_tokens]`` (PAD_SLOT_ID = -1 is skipped).
        cache_c: latent paged cache ``[num_blocks, block_size, 1, kv_lora_rank]``.
        cache_pe: rope paged cache ``[num_blocks, block_size, 1, rope_dim]``.
        gate_out: output buffer ``[num_tokens, num_heads * head_dim]``.
    Returns:
        (q_pe_rope [n, H, rope_dim], q_nope [n, H, head_dim-rope_dim],
         kv_c_ret [n, kv_lora_rank], k_pe_ret [n, rope_dim]).
    """
    assert qg.dtype == torch.bfloat16, "fused_mla_rope_scatter_split only supports bf16"
    num_tokens = qg.shape[0]
    nope_dim = head_dim - rope_dim
    device = qg.device
    dtype = qg.dtype

    q_pe = torch.empty(num_tokens, num_heads, rope_dim, device=device, dtype=dtype)
    q_nope = torch.empty(num_tokens, num_heads, nope_dim, device=device, dtype=dtype)
    kv_c_ret = torch.empty(num_tokens, kv_lora_rank, device=device, dtype=dtype)
    k_pe_ret = torch.empty(num_tokens, rope_dim, device=device, dtype=dtype)
    if num_tokens == 0:
        return q_pe, q_nope, kv_c_ret, k_pe_ret

    init_device_properties_triton()
    core_num = get_vectorcore_num()

    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num
    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num
    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num
    num_tokens_each_tail_core = num_tokens // core_num

    total_core = front_core_num + tail_core_num
    block_dim = core_num
    if total_core < core_num:
        block_dim = total_core

    fused_mla_rope_scatter_split_kernel[(block_dim,)](
        qg,
        q_weight,
        kv576,
        cos,
        sin,
        slots,
        cache_c,
        cache_pe,
        gate_out,
        q_pe,
        q_nope,
        kv_c_ret,
        k_pe_ret,
        num_tokens,
        front_core_num,
        num_tokens_each_front_core,
        num_tokens_each_tail_core,
        cache_c.shape[1],
        num_heads,
        head_dim,
        rope_dim,
        kv_lora_rank,
        eps,
    )
    return q_pe, q_nope, kv_c_ret, k_pe_ret


def triton_fused_mla_rope_scatter_split_fake(
    qg: torch.Tensor,
    q_weight: torch.Tensor,
    kv576: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    cache_c: torch.Tensor,
    cache_pe: torch.Tensor,
    gate_out: torch.Tensor,
    num_heads: int,
    kv_lora_rank: int,
    rope_dim: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qg.shape[0]
    nope_dim = head_dim - rope_dim
    device = qg.device
    dtype = qg.dtype
    q_pe = torch.empty(num_tokens, num_heads, rope_dim, device=device, dtype=dtype)
    q_nope = torch.empty(num_tokens, num_heads, nope_dim, device=device, dtype=dtype)
    kv_c_ret = torch.empty(num_tokens, kv_lora_rank, device=device, dtype=dtype)
    k_pe_ret = torch.empty(num_tokens, rope_dim, device=device, dtype=dtype)
    return q_pe, q_nope, kv_c_ret, k_pe_ret


direct_register_custom_op(
    op_name="triton_fused_knorm_rearrange",
    op_func=triton_fused_knorm_rearrange,
    fake_impl=triton_fused_knorm_rearrange_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="triton_fused_mla_rope_scatter_split",
    op_func=triton_fused_mla_rope_scatter_split,
    fake_impl=triton_fused_mla_rope_scatter_split_fake,
    mutates_args=["cache_c", "cache_pe", "gate_out"],
    dispatch_key="PrivateUse1",
)
