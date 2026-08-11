// True fused RoPE + paged-scatter for MLA when qk_latent_layernorm=False
// (no RMSNorm). Adapted verbatim from CANN's kv_rms_norm_rope_cache_b16_pa.h:
// the k_pe RoPE path (GatherMask-based interleave rotation) and the paged
// scatter (ScatterUpdatePABnsd) are copied unchanged; only the c_kv path's
// RMSNorm is dropped (replaced by a Cast identity so the TQue outQueue still
// establishes MTE2->V->MTE3 sync). Single kernel => no intermediate k_pe
// tensor => no cross-stream read-before-write race on large multi-request
// prefills (which the split npu_interleave_rope + scatter approach hit).

#define __aicore__ [aicore]
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../op_host/kv_rope_cache_tiling.h"

using namespace AscendC;
using namespace kv_rope_cache_op;

constexpr int64_t PA_BNSD_NO_QUANT = 3;

template <typename KV_DTYPE>
class KernelKvRopeCacheB16PA {
public:
    struct Tiling {
        int64_t batchSize, seqLength, numHead, blockFactor, ubFactor, numBlocks;
        int64_t isOutputKv, rmsNormLength, ropeLength;
    };

    __aicore__ inline KernelKvRopeCacheB16PA(TPipe* pipe, const Tiling& t)
        : pipe_(pipe), t_(t) {}

    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR cos, GM_ADDR sin, GM_ADDR index,
        GM_ADDR k_cache, GM_ADDR v_cache,
        GM_ADDR optional_k_rope, GM_ADDR optional_c_kv) {
        RMS_NORM_LENGTH = t_.rmsNormLength;
        ROPE_LENGTH = t_.ropeLength;
        // Safe per-block token count: idle blocks with myStart >= total get 0
        // (the CANN last-block formula goes negative & OOBs when total is not
        // a multiple of blockFactor * numBlocks).
        int64_t myStart = GetBlockIdx() * t_.blockFactor;
        int64_t remaining = t_.batchSize * t_.seqLength * t_.numHead - myStart;
        int64_t currentBlockFactor = remaining > 0
            ? (remaining < t_.blockFactor ? remaining : t_.blockFactor) : 0;
        ubFactor = t_.ubFactor;
        ubLoop = currentBlockFactor / ubFactor;
        ubTail = currentBlockFactor - ubLoop * ubFactor;
        isOutputKv = t_.isOutputKv != 0;

        kvGm.SetGlobalBuffer(
            (__gm__ KV_DTYPE*)kv + GetBlockIdx() * t_.blockFactor *
            (RMS_NORM_LENGTH + ROPE_LENGTH));
        cosGm.SetGlobalBuffer(
            (__gm__ KV_DTYPE*)cos + GetBlockIdx() * t_.blockFactor * ROPE_LENGTH);
        sinGm.SetGlobalBuffer(
            (__gm__ KV_DTYPE*)sin + GetBlockIdx() * t_.blockFactor * ROPE_LENGTH);
        indexGm.SetGlobalBuffer((__gm__ int64_t*)index);
        kCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)k_cache);   // rope cache (64)
        vCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)v_cache);    // nope cache (512)
        if (isOutputKv) {
            kCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE*)optional_k_rope);
            vCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE*)optional_c_kv);
        }

        // init pipe
        pipe_->InitBuffer(inQueueX, 2, ubFactor * (RMS_NORM_LENGTH + ROPE_LENGTH) * sizeof(KV_DTYPE));
        pipe_->InitBuffer(outQueue, 2, ubFactor * RMS_NORM_LENGTH * sizeof(KV_DTYPE));
        pipe_->InitBuffer(wsBuffer, 3 * ubFactor * RMS_NORM_LENGTH * sizeof(float));
        // dedicated fp32 workspace for the c_kv Cast-identity V bridge (must
        // NOT alias the RoPE workspace, else the c_kv path produces zeros).
        pipe_->InitBuffer(cKvWsBuf, ubFactor * RMS_NORM_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process() {
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockCount = 1;
        LocalTensor<float> workspaceBuffer = wsBuffer.template Get<float>();

        for (int64_t loopIdx = 0; loopIdx < ubLoop; ++loopIdx) {
            ProcessUbLoop(loopIdx, ubFactor, padParams, copyParamsContinguous, workspaceBuffer);
        }
        if (ubTail > 0) {
            ProcessUbLoop(ubLoop, ubTail, padParams, copyParamsContinguous, workspaceBuffer);
        }
    }

private:
    __aicore__ inline void ProcessUbLoop(
        int64_t loopIdx, int64_t rows,
        DataCopyPadExtParams<KV_DTYPE>& padParams,
        DataCopyExtParams& copyParamsContinguous,
        LocalTensor<float>& workspaceBuffer) {
        int64_t kvOff = loopIdx * ubFactor * (RMS_NORM_LENGTH + ROPE_LENGTH);
        int64_t freqOff = loopIdx * ubFactor * ROPE_LENGTH;
        int64_t startIdx = GetBlockIdx() * t_.blockFactor + loopIdx * ubFactor;

        // ---------- k_pe path: CopyIn rope part (64) + cos + sin, RoPE, scatter ----------
        LocalTensor<KV_DTYPE> ropeLocal = inQueueX.template AllocTensor<KV_DTYPE>();
        LocalTensor<KV_DTYPE> cosLocal = ropeLocal[rows * ROPE_LENGTH];
        LocalTensor<KV_DTYPE> sinLocal = cosLocal[rows * ROPE_LENGTH];
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(ROPE_LENGTH * sizeof(KV_DTYPE)),
            static_cast<uint32_t>(RMS_NORM_LENGTH * sizeof(KV_DTYPE)), 0, 0};
        DataCopyPad(ropeLocal, kvGm[kvOff + RMS_NORM_LENGTH], copyParams, padParams);
        copyParamsContinguous.blockLen = rows * ROPE_LENGTH * sizeof(KV_DTYPE);
        DataCopyPad(cosLocal, cosGm[freqOff], copyParamsContinguous, padParams);
        DataCopyPad(sinLocal, sinGm[freqOff], copyParamsContinguous, padParams);
        inQueueX.EnQue(ropeLocal);
        ropeLocal = inQueueX.template DeQue<KV_DTYPE>();
        cosLocal = ropeLocal[rows * ROPE_LENGTH];
        sinLocal = cosLocal[rows * ROPE_LENGTH];

        LocalTensor<KV_DTYPE> ropeOut = outQueue.template AllocTensor<KV_DTYPE>();
        RoPE<KV_DTYPE, true>(ropeOut, ropeLocal, cosLocal, sinLocal, workspaceBuffer, rows, ROPE_LENGTH);
        inQueueX.FreeTensor(ropeLocal);
        outQueue.EnQue(ropeOut);
        ropeOut = outQueue.template DeQue<KV_DTYPE>();
        ScatterUpdatePABnsd<KV_DTYPE>(kCacheGm, kCacheGmNd, ropeOut, startIdx, rows, ROPE_LENGTH);
        outQueue.FreeTensor(ropeOut);

        // ---------- c_kv path: CopyIn nope part (512), Cast identity (V), scatter (NO RMSNorm) ----------
        LocalTensor<KV_DTYPE> xLocal = inQueueX.template AllocTensor<KV_DTYPE>();
        DataCopyExtParams copyParamsNope{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
            static_cast<uint32_t>(ROPE_LENGTH * sizeof(KV_DTYPE)), 0, 0};
        DataCopyPad(xLocal, kvGm[kvOff], copyParamsNope, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.template DeQue<KV_DTYPE>();

        // V identity: out = x (bf16 -> fp32 -> bf16) via a dedicated workspace
        // (not the RoPE wsBuffer) so outQueue establishes MTE2->V->MTE3 sync.
        LocalTensor<float> cKvWsFp32 = cKvWsBuf.template Get<float>();
        LocalTensor<KV_DTYPE> nopeOut = outQueue.template AllocTensor<KV_DTYPE>();
        Cast(cKvWsFp32, xLocal, RoundMode::CAST_NONE, rows * RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();
        Cast(nopeOut, cKvWsFp32, RoundMode::CAST_RINT, rows * RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();
        inQueueX.FreeTensor(xLocal);
        outQueue.EnQue(nopeOut);
        nopeOut = outQueue.template DeQue<KV_DTYPE>();
        ScatterUpdatePABnsd<KV_DTYPE>(vCacheGm, vCacheGmNd, nopeOut, startIdx, rows, RMS_NORM_LENGTH);
        outQueue.FreeTensor(nopeOut);
    }

    // ---- RoPE: verbatim from CANN kv_rms_norm_rope_cache_b16_pa.h ----
    template <typename T, bool isElementWise = true>
    __aicore__ inline void RoPE(
        const LocalTensor<T>& outLocal, const LocalTensor<T>& xLocal, const LocalTensor<T>& cosLocal,
        const LocalTensor<T>& sinLocal, const LocalTensor<float>& wsLocal, int64_t rows, int64_t headSize) {
        constexpr static int64_t NUM_ONE = 1;
        constexpr static int64_t NUM_TWO = 2;
        constexpr static int64_t NUM_FOUR = 4;
        constexpr static int64_t NUM_EIGHT = 8;
        if constexpr (isElementWise) {
            int64_t cosLocalFp32Offset = rows * headSize * 0;
            int64_t sinLocalFp32Offset = rows * headSize * 1;
            int64_t y0Offset = rows * headSize * 2;
            int64_t y1Offset = rows * headSize * 3;
            int64_t realLocalFp32Offset = rows * headSize * 4;
            int64_t imagLocalFp32Offset = rows * headSize * 5;
            int64_t realLocalOffset = rows * headSize * 6;
            int64_t imagLocalOffset = rows * headSize * 7;
            LocalTensor<float> cosLocalFp32 = wsLocal[cosLocalFp32Offset];
            LocalTensor<float> sinLocalFp32 = wsLocal[sinLocalFp32Offset];
            LocalTensor<float> y0 = wsLocal[y0Offset];
            LocalTensor<float> y1 = wsLocal[y1Offset];
            LocalTensor<float> realLocalFp32 = wsLocal[realLocalFp32Offset];
            LocalTensor<float> imagLocalFp32 = wsLocal[imagLocalFp32Offset];
            LocalTensor<T> realLocal = wsLocal[realLocalOffset].template ReinterpretCast<T>();
            LocalTensor<T> imagLocal = wsLocal[imagLocalOffset].template ReinterpretCast<T>();
            Cast(cosLocalFp32, cosLocal, RoundMode::CAST_NONE, rows * headSize);
            Cast(sinLocalFp32, sinLocal, RoundMode::CAST_NONE, rows * headSize);
            PipeBarrier<PIPE_V>();
            uint64_t rsvdCnt = 0;
            GatherMask(realLocal, xLocal, NUM_ONE, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            GatherMask(imagLocal, xLocal, NUM_TWO, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();
            Cast(realLocalFp32, realLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            Cast(imagLocalFp32, imagLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();
            Mul(y0, realLocalFp32, cosLocalFp32, (headSize >> 1), rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y0[(headSize >> 1)], imagLocalFp32, cosLocalFp32[(headSize >> 1)], (headSize >> 1), rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();
            Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();
            Mul(y1, imagLocalFp32, sinLocalFp32, (headSize >> 1), rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y1[(headSize >> 1)], realLocalFp32, sinLocalFp32[(headSize >> 1)], (headSize >> 1), rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();
            Add(y0, y0, y1, rows * headSize);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, y0, RoundMode::CAST_RINT, rows * headSize);
            } else if constexpr (std::is_same<T, half>::value) {
                Cast(outLocal, y0, RoundMode::CAST_NONE, rows * headSize);
            }
        }
    }

    // ---- ScatterUpdatePABnsd: verbatim from CANN ----
    template <typename T>
    __aicore__ inline void ScatterUpdatePABnsd(
        const GlobalTensor<T>& dst, const GlobalTensor<T>& dstNd, const LocalTensor<T>& outLocal, int64_t startIdx,
        int64_t rows, int64_t headSize) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = headSize * i;
            if (isOutputKv) {
                int64_t gmOffsetNd = tokenId * headSize;
                DataCopyPad(dstNd[gmOffsetNd], outLocal[ubOffset], copyParams);
            }
            int64_t tokensPerBatch = t_.seqLength * t_.numHead;
            int64_t batchId = tokenId / tokensPerBatch;
            int64_t tokenInBatch = tokenId % tokensPerBatch;
            int64_t headIdx = tokenInBatch / t_.seqLength;
            int64_t seqIdx = tokenInBatch % t_.seqLength;
            int64_t offset = indexGm(batchId * t_.seqLength + seqIdx);
            if (offset >= 0) {
                int64_t gmOffset = offset * headSize * t_.numHead + headIdx * headSize;
                SToMTE3Sync();
                DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
            }
        }
    }

    __aicore__ inline void SToMTE3Sync() {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(ev);
        WaitFlag<HardEvent::S_MTE3>(ev);
    }

private:
    TPipe* pipe_ = nullptr;
    Tiling t_;

    GlobalTensor<KV_DTYPE> kvGm, cosGm, sinGm;
    GlobalTensor<KV_DTYPE> kCacheGm, vCacheGm, kCacheGmNd, vCacheGmNd;
    GlobalTensor<int64_t> indexGm;

    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> wsBuffer;
    TBuf<TPosition::VECCALC> cKvWsBuf;

    int64_t ubFactor = 8;
    int64_t ubLoop = 1;
    int64_t ubTail = 0;
    int64_t RMS_NORM_LENGTH = 512;
    int64_t ROPE_LENGTH = 64;
    bool isOutputKv = false;
};

extern "C" __global__ __aicore__ void kv_rope_cache(
    GM_ADDR kv, GM_ADDR cos, GM_ADDR sin, GM_ADDR index,
    GM_ADDR k_cache, GM_ADDR v_cache,
    GM_ADDR k_rope_out, GM_ADDR c_kv_out,
    int64_t batchSize, int64_t seqLength, int64_t numHead,
    int64_t blockFactor, int64_t ubFactor, int64_t numBlocks,
    int64_t isOutputKv, int64_t rmsNormLength, int64_t ropeLength) {
    TPipe pipe;
    KernelKvRopeCacheB16PA<bfloat16_t>::Tiling t{
        batchSize, seqLength, numHead, blockFactor, ubFactor, numBlocks,
        isOutputKv, rmsNormLength, ropeLength};
    KernelKvRopeCacheB16PA<bfloat16_t> op(&pipe, t);
    op.Init(kv, cos, sin, index, k_cache, v_cache, k_rope_out, c_kv_out);
    op.Process();
}

namespace vllm_ascend {

extern void kv_rope_cache_impl(
    void* stream, void* gm_kv, void* gm_cos, void* gm_sin, void* gm_index,
    void* gm_k_cache, void* gm_v_cache, void* gm_k_rope_out, void* gm_c_kv_out,
    int64_t batchSize, int64_t seqLength, int64_t numHead,
    int64_t blockFactor, int64_t ubFactor, int64_t numBlocks,
    int64_t isOutputKv, int64_t rmsNormLength, int64_t ropeLength,
    const uint32_t block_dim) {
    kv_rope_cache<<<block_dim, nullptr, stream>>>(
        gm_kv, gm_cos, gm_sin, gm_index, gm_k_cache, gm_v_cache,
        gm_k_rope_out, gm_c_kv_out,
        batchSize, seqLength, numHead, blockFactor, ubFactor, numBlocks,
        isOutputKv, rmsNormLength, ropeLength);
}

}  // namespace vllm_ascend
