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
 * Qwen36KvNormPermute: fused Gemma-RMSNorm(k) + [rot|pass|v] permute for the
 * Qwen3.6 MLA KV projection.
 *
 * Input kv:   [numRow, 1024] bf16, layout [k_h0(256) k_h1(256) v_h0(256) v_h1(256)]
 *             (the merged k_proj/v_proj GEMM output).
 * Input gamma: [256] bf16 (k_norm weight; Gemma style, applied as (1 + w)).
 * Attr epsilon: variance epsilon.
 * Output y:   [numRow, 1024] bf16, layout
 *             [rot_h0(64) rot_h1(64) pass_h0(192) pass_h1(192) v(512)]
 *             (the kv_a_proj input order).
 *
 * Per head h: y_h = x_h * rsqrt(mean(x_h^2) + eps) * (1 + gamma)
 * computed with fp32 accumulation (matching vllm GemmaRMSNorm semantics).
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t KV_ROW_DIM = 1024;
constexpr uint32_t K_DIM = 512;          // 2 heads * 256
constexpr uint32_t HEAD_DIM = 256;
constexpr uint32_t ROT_DIM = 64;
constexpr uint32_t PASS_DIM = HEAD_DIM - ROT_DIM;  // 192
constexpr uint32_t V_OFFSET = K_DIM;     // 512
constexpr uint32_t NUM_PER_REP_FP32 = 64;
constexpr uint32_t ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t ONE_BLK_SIZE = 32;
constexpr float MINUS_ZERO = -0.0f;

// Whole-row sum of a <=512-element fp32 tensor into dst[0]. Adapted from
// add_rms_norm_bias's ReduceSumFP32 (proven on dav-c220-vec / 910B).
__aicore__ inline void ReduceSumRow(LocalTensor<float> &dst, LocalTensor<float> &src,
                                    LocalTensor<float> &work, int32_t count)
{
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work, 0.f, NUM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    if (repeatTimes > 0) {
        Add(work, src, work, mask, static_cast<uint8_t>(repeatTimes), repeatParams);
        PipeBarrier<PIPE_V>();
    }
    if (tailCount != 0) {
        Add(work, src[bodyCount], work, tailCount, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
    WholeReduceSum<float, false>(dst, work, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
}

template <typename T>
class Qwen36KvNormPermuteKernel {
public:
    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR gamma, GM_ADDR y,
                                const Qwen36KvNormPermuteTilingData &tiling, TPipe *pipe,
                                float epsilon)
    {
        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kv), tiling.numRow * KV_ROW_DIM);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma), HEAD_DIM);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), tiling.numRow * KV_ROW_DIM);
        pipe_ = pipe;
        numRow_ = static_cast<uint64_t>(tiling.numRow);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        epsilon_ = epsilon;
        avgFactor_ = 1.0f / 256.0f;

        pipe_->InitBuffer(inQue_, 2, KV_ROW_DIM * sizeof(T));
        pipe_->InitBuffer(outQue_, 2, KV_ROW_DIM * sizeof(T));
        pipe_->InitBuffer(gammaQue_, 1, HEAD_DIM * sizeof(T));
        pipe_->InitBuffer(kF32Buf_, K_DIM * sizeof(float));
        pipe_->InitBuffer(sqxBuf_, HEAD_DIM * sizeof(float));
        pipe_->InitBuffer(redBuf_, NUM_PER_REP_FP32 * sizeof(float));
        pipe_->InitBuffer(gammaP1Buf_, HEAD_DIM * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LoadGamma();
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        for (uint64_t row = coreIdx; row < numRow_; row += usedCoreNum_) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void LoadGamma()
    {
        {
            LocalTensor<T> g = gammaQue_.AllocTensor<T>();
            DataCopy(g, gammaGm_, HEAD_DIM);
            gammaQue_.EnQue(g);
        }
        LocalTensor<T> gammaLocal = gammaQue_.DeQue<T>();
        LocalTensor<float> gammaF32 = gammaP1Buf_.Get<float>();
        Cast(gammaF32, gammaLocal, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Adds(gammaF32, gammaF32, 1.0f, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        gammaQue_.FreeTensor(gammaLocal);
    }

    __aicore__ inline float ComputeRstd(LocalTensor<float> &x, LocalTensor<float> &sqx,
                                        LocalTensor<float> &red)
    {
        Mul(sqx, x, x, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor_, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        ReduceSumRow(sqx, sqx, red, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Adds(sqx, sqx, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqx, sqx, 1);
        Duplicate(red, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, red, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqx.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        return rstdValue;
    }

    __aicore__ inline void ProcessRow(uint64_t row)
    {
        {
            LocalTensor<T> xLocal = inQue_.AllocTensor<T>();
            DataCopy(xLocal, kvGm_[row * KV_ROW_DIM], KV_ROW_DIM);
            inQue_.EnQue(xLocal);
        }
        LocalTensor<T> xLocal = inQue_.DeQue<T>();
        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();

        LocalTensor<float> kF32 = kF32Buf_.Get<float>();
        LocalTensor<float> sqx = sqxBuf_.Get<float>();
        LocalTensor<float> red = redBuf_.Get<float>();
        LocalTensor<float> gammaF32 = gammaP1Buf_.Get<float>();

        // --- head 0 ---
        Cast(kF32, xLocal, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        float rstd0 = ComputeRstd(kF32, sqx, red);
        Muls(kF32, kF32, rstd0, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Mul(kF32, kF32, gammaF32, HEAD_DIM);
        PipeBarrier<PIPE_V>();

        // --- head 1 ---
        LocalTensor<float> kF32H1 = kF32[HEAD_DIM];
        Cast(kF32H1, xLocal[HEAD_DIM], RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        float rstd1 = ComputeRstd(kF32H1, sqx, red);
        Muls(kF32H1, kF32H1, rstd1, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Mul(kF32H1, kF32H1, gammaF32, HEAD_DIM);
        PipeBarrier<PIPE_V>();

        // --- permuted write: [rot_h0 | rot_h1 | pass_h0 | pass_h1 | v] ---
        Cast(yLocal, kF32, RoundMode::CAST_RINT, ROT_DIM);
        PipeBarrier<PIPE_V>();
        Cast(yLocal[ROT_DIM], kF32H1, RoundMode::CAST_RINT, ROT_DIM);
        PipeBarrier<PIPE_V>();
        Cast(yLocal[2 * ROT_DIM], kF32[ROT_DIM], RoundMode::CAST_RINT, PASS_DIM);
        PipeBarrier<PIPE_V>();
        Cast(yLocal[2 * ROT_DIM + PASS_DIM], kF32H1[ROT_DIM], RoundMode::CAST_RINT, PASS_DIM);
        PipeBarrier<PIPE_V>();
        DataCopy(yLocal[V_OFFSET], xLocal[V_OFFSET], KV_ROW_DIM - V_OFFSET);
        PipeBarrier<PIPE_V>();

        inQue_.FreeTensor(xLocal);
        outQue_.EnQue<T>(yLocal);
        LocalTensor<T> yOut = outQue_.DeQue<T>();
        DataCopy(yGm_[row * KV_ROW_DIM], yOut, KV_ROW_DIM);
        outQue_.FreeTensor(yOut);
    }

private:
    TPipe *pipe_ = nullptr;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;
    TQue<TPosition::VECIN, 1> gammaQue_;
    TBuf<TPosition::VECCALC> kF32Buf_;
    TBuf<TPosition::VECCALC> sqxBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    TBuf<TPosition::VECCALC> gammaP1Buf_;
    GlobalTensor<T> kvGm_;
    GlobalTensor<T> gammaGm_;
    GlobalTensor<T> yGm_;
    uint64_t numRow_ = 0;
    uint64_t usedCoreNum_ = 1;
    float epsilon_ = 1e-6f;
    float avgFactor_ = 0.f;
};
} // namespace

extern "C" __global__ __aicore__ void qwen36_kv_norm_permute(
    GM_ADDR kv, GM_ADDR gamma, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        Qwen36KvNormPermuteKernel<bfloat16_t> op;
        op.Init(kv, gamma, y, tilingData, &pipe, tilingData.epsilon);
        op.Process();
    }
}
