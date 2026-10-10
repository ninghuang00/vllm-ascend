/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You can obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*
 * Qwen36KvRopeScatter: norm-free variant of npu_kv_rmsnorm_rope_cache for the
 * Qwen3.6 MLA decode path — fused split + NeoX-half RoPE + dual paged-cache
 * scatter in one kernel.
 *
 * Input kv:         [numRow, 576] bf16, layout [ckv(512) | k_pe(64)]
 *                   (the kv_a_proj_with_mqa GEMM output).
 * Input cos/sin:    [numRow, 64] bf16, per-token NeoX rotary rows (already
 *                   position-gathered by the caller).
 * Input slots:      [numRow] int64, flat slot ids (block_idx * blockSize + off).
 * InOut ckv_cache:  [numBlocks, blockSize, 1, 512] bf16 (Norm layout).
 * InOut kpe_cache:  [numBlocks, blockSize, 1, 64] bf16 (Norm layout).
 *
 * Per row r, slot = slots[r], bi = slot / blockSize, bo = slot % blockSize:
 *   pe[i] = kv[r][512+i] * cos[i] + rotate_half(kv[r][512:576])[i] * sin[i]
 *           (fp32 accumulate, round-to-nearest-even to bf16 — bit-exact with
 *           torch_npu.npu_rotary_mul(rotary_mode="half"))
 *   ckv_cache[bi, bo, 0, :] = kv[r][0:512]   (bit-exact passthrough)
 *   kpe_cache[bi, bo, 0, :] = pe
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t KV_ROW_DIM = 576;
constexpr uint32_t CKV_DIM = 512;
constexpr uint32_t PE_DIM = 64;
constexpr uint32_t PE_HALF_DIM = 32;

template <typename T>
class Qwen36KvRopeScatterKernel {
public:
    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR cos, GM_ADDR sin, GM_ADDR slots,
                                GM_ADDR ckvCacheIn, GM_ADDR kpeCacheIn,
                                GM_ADDR ckvCacheOut, GM_ADDR kpeCacheOut,
                                const Qwen36KvRopeScatterTilingData &tiling, TPipe *pipe)
    {
        (void)ckvCacheIn;
        (void)kpeCacheIn;
        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kv), tiling.numRow * KV_ROW_DIM);
        cosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cos), tiling.numRow * PE_DIM);
        sinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(sin), tiling.numRow * PE_DIM);
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slots), tiling.numRow);
        // The in-place caches are declared as same-name Input+Output; the
        // generated wrapper passes both pointers (they alias at runtime).
        // Writes go through the output pointers.
        ckvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ckvCacheOut),
                                    static_cast<uint64_t>(tiling.numBlock) * tiling.blockSize * CKV_DIM);
        kpeCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kpeCacheOut),
                                    static_cast<uint64_t>(tiling.numBlock) * tiling.blockSize * PE_DIM);
        pipe_ = pipe;
        numRow_ = static_cast<uint64_t>(tiling.numRow);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);

        pipe_->InitBuffer(inQue_, 2, KV_ROW_DIM * sizeof(T));
        pipe_->InitBuffer(cosQue_, 2, PE_DIM * sizeof(T));
        pipe_->InitBuffer(sinQue_, 2, PE_DIM * sizeof(T));
        pipe_->InitBuffer(outQue_, 2, KV_ROW_DIM * sizeof(T));
        pipe_->InitBuffer(peF32Buf_, PE_DIM * sizeof(float));
        pipe_->InitBuffer(cosF32Buf_, PE_DIM * sizeof(float));
        pipe_->InitBuffer(sinF32Buf_, PE_DIM * sizeof(float));
        pipe_->InitBuffer(t1F32Buf_, PE_DIM * sizeof(float));
        pipe_->InitBuffer(rF32Buf_, PE_HALF_DIM * sizeof(float));
        pipe_->InitBuffer(outF32Buf_, PE_DIM * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        for (uint64_t row = coreIdx; row < numRow_; row += usedCoreNum_) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint64_t row)
    {
        // slot -> (block, offset); flat slot semantics identical to
        // npu_scatter_pa_kv_cache / the index_put fallback chain.
        int64_t slot = slotGm_.GetValue(static_cast<int64_t>(row));
        uint64_t bi = static_cast<uint64_t>(slot) / blockSize_;
        uint64_t bo = static_cast<uint64_t>(slot) % blockSize_;
        uint64_t cacheRow = bi * blockSize_ + bo;

        LocalTensor<T> xLocal;
        LocalTensor<T> cosLocal;
        LocalTensor<T> sinLocal;
        {
            xLocal = inQue_.AllocTensor<T>();
            DataCopy(xLocal, kvGm_[row * KV_ROW_DIM], KV_ROW_DIM);
            inQue_.EnQue(xLocal);

            cosLocal = cosQue_.AllocTensor<T>();
            DataCopy(cosLocal, cosGm_[row * PE_DIM], PE_DIM);
            cosQue_.EnQue(cosLocal);

            sinLocal = sinQue_.AllocTensor<T>();
            DataCopy(sinLocal, sinGm_[row * PE_DIM], PE_DIM);
            sinQue_.EnQue(sinLocal);
        }
        xLocal = inQue_.DeQue<T>();
        cosLocal = cosQue_.DeQue<T>();
        sinLocal = sinQue_.DeQue<T>();

        LocalTensor<float> peF32 = peF32Buf_.Get<float>();
        LocalTensor<float> cosF32 = cosF32Buf_.Get<float>();
        LocalTensor<float> sinF32 = sinF32Buf_.Get<float>();
        LocalTensor<float> t1F32 = t1F32Buf_.Get<float>();
        LocalTensor<float> rF32 = rF32Buf_.Get<float>();
        LocalTensor<float> outF32 = outF32Buf_.Get<float>();

        // fp32 casts of the rope inputs.
        Cast(peF32, xLocal[CKV_DIM], RoundMode::CAST_NONE, PE_DIM);
        PipeBarrier<PIPE_V>();
        Cast(cosF32, cosLocal, RoundMode::CAST_NONE, PE_DIM);
        PipeBarrier<PIPE_V>();
        Cast(sinF32, sinLocal, RoundMode::CAST_NONE, PE_DIM);
        PipeBarrier<PIPE_V>();

        // t1 = x * cos
        Mul(t1F32, peF32, cosF32, PE_DIM);
        PipeBarrier<PIPE_V>();

        // Segment A (i < 32): out[i] = x[i]*cos[i] - x[i+32]*sin[i]
        Mul(rF32, peF32[PE_HALF_DIM], sinF32, PE_HALF_DIM);
        PipeBarrier<PIPE_V>();
        Sub(outF32, t1F32, rF32, PE_HALF_DIM);
        PipeBarrier<PIPE_V>();

        // Segment B (i >= 32): out[i] = x[i]*cos[i] + x[i-32]*sin[i]
        Mul(rF32, peF32, sinF32[PE_HALF_DIM], PE_HALF_DIM);
        PipeBarrier<PIPE_V>();
        Add(outF32[PE_HALF_DIM], t1F32[PE_HALF_DIM], rF32, PE_HALF_DIM);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
        // ckv passthrough: ub-to-ub copy (proven in qwen36_kv_norm_permute).
        DataCopy(yLocal, xLocal, CKV_DIM);
        // roped k_pe, fp32 -> bf16 round-to-nearest-even.
        Cast(yLocal[CKV_DIM], outF32, RoundMode::CAST_RINT, PE_DIM);
        PipeBarrier<PIPE_V>();

        inQue_.FreeTensor(xLocal);
        cosQue_.FreeTensor(cosLocal);
        sinQue_.FreeTensor(sinLocal);
        outQue_.EnQue<T>(yLocal);
        LocalTensor<T> yOut = outQue_.DeQue<T>();
        DataCopy(ckvCacheGm_[cacheRow * CKV_DIM], yOut, CKV_DIM);
        DataCopy(kpeCacheGm_[cacheRow * PE_DIM], yOut[CKV_DIM], PE_DIM);
        outQue_.FreeTensor(yOut);
    }

private:
    TPipe *pipe_ = nullptr;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECIN, 2> cosQue_;
    TQue<TPosition::VECIN, 2> sinQue_;
    TQue<TPosition::VECOUT, 2> outQue_;
    TBuf<TPosition::VECCALC> peF32Buf_;
    TBuf<TPosition::VECCALC> cosF32Buf_;
    TBuf<TPosition::VECCALC> sinF32Buf_;
    TBuf<TPosition::VECCALC> t1F32Buf_;
    TBuf<TPosition::VECCALC> rF32Buf_;
    TBuf<TPosition::VECCALC> outF32Buf_;
    GlobalTensor<T> kvGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<int64_t> slotGm_;
    GlobalTensor<T> ckvCacheGm_;
    GlobalTensor<T> kpeCacheGm_;
    uint64_t numRow_ = 0;
    uint64_t usedCoreNum_ = 1;
    uint64_t blockSize_ = 1;
};
} // namespace

extern "C" __global__ __aicore__ void qwen36_kv_rope_scatter(
    GM_ADDR kv, GM_ADDR cos, GM_ADDR sin, GM_ADDR slots, GM_ADDR ckv_cache, GM_ADDR kpe_cache,
    GM_ADDR ckv_cache_out, GM_ADDR kpe_cache_out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        Qwen36KvRopeScatterKernel<bfloat16_t> op;
        op.Init(kv, cos, sin, slots, ckv_cache, kpe_cache, ckv_cache_out, kpe_cache_out, tilingData, &pipe);
        op.Process();
    }
}
