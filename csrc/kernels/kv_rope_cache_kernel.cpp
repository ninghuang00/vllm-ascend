#include "kernel_operator.h"
#include <stdio.h>
#include "types.h"
#include "utils.h"

using vllm_ascend::AccType;
using vllm_ascend::local_mem_copy;

template <typename T> struct KernelAccType { using type = float; };
template <> struct KernelAccType<float> { using type = float; };

template <typename scalar_t> class KvRopeCache {
    using acc_t = typename KernelAccType<scalar_t>::type;
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;
    using local_acc_t = AscendC::LocalTensor<acc_t>;

public:
    __aicore__ inline KvRopeCache() {}

    __aicore__ inline void init(
        __gm__ scalar_t* kv, __gm__ scalar_t* cos, __gm__ scalar_t* sin,
        __gm__ int64_t* slots, __gm__ scalar_t* k_cache, __gm__ scalar_t* ckv_cache,
        __gm__ scalar_t* k_pe_out, __gm__ scalar_t* k_nope_out,
        const int kvLoraRank, const int ropeDim, const int64_t numTokens,
        const bool isOutputKv, AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        kvGm_.SetGlobalBuffer(kv);
        cosGm_.SetGlobalBuffer(cos);
        sinGm_.SetGlobalBuffer(sin);
        slotsGm_.SetGlobalBuffer(slots);
        kCacheGm_.SetGlobalBuffer(k_cache);
        ckvCacheGm_.SetGlobalBuffer(ckv_cache);
        kPeOutGm_.SetGlobalBuffer(k_pe_out);
        kNopeOutGm_.SetGlobalBuffer(k_nope_out);

        kvLoraRank_ = kvLoraRank;
        ropeDim_ = ropeDim;
        numTokens_ = numTokens;
        isOutputKv_ = isOutputKv;
        totalDim_ = kvLoraRank + ropeDim;
        halfRopeDim_ = ropeDim / 2;

        const int scalarSize = sizeof(scalar_t);
        const int accSize = sizeof(acc_t);

        pipe_->InitBuffer(inKvQue_, 1, totalDim_ * scalarSize);
        pipe_->InitBuffer(inCosQue_, 1, ropeDim_ * scalarSize);
        pipe_->InitBuffer(inSinQue_, 1, ropeDim_ * scalarSize);
        pipe_->InitBuffer(outPeQue_, 1, ropeDim_ * scalarSize);
        pipe_->InitBuffer(outNopeQue_, 1, kvLoraRank_ * scalarSize);

        // Compute offsets for calc buffer (all in acc_t = float)
        cosOff_ = 0;
        sinOff_ = cosOff_ + ropeDim_ * accSize;
        peOff_ = sinOff_ + ropeDim_ * accSize;
        nopeOff_ = peOff_ + ropeDim_ * accSize;
        tmp0Off_ = nopeOff_ + kvLoraRank_ * accSize;
        tmp1Off_ = tmp0Off_ + ropeDim_ * accSize;
        int calcBufSize = tmp1Off_ + ropeDim_ * accSize;
        pipe_->InitBuffer(calcBuf_, calcBufSize);
    }

    __aicore__ inline void process_token(int64_t tokenIdx)
    {
        int64_t kvOffset = tokenIdx * totalDim_;
        int64_t cosOffset = tokenIdx * ropeDim_;

        // Load kv from GM
        local_scalar_t kvLocal = inKvQue_.template AllocTensor<scalar_t>();
        AscendC::DataCopy(kvLocal, kvGm_[kvOffset], totalDim_);
        inKvQue_.EnQue(kvLocal);
        kvLocal = inKvQue_.template DeQue<scalar_t>();

        // Load cos/sin from GM
        local_scalar_t cosLocal = inCosQue_.template AllocTensor<scalar_t>();
        AscendC::DataCopy(cosLocal, cosGm_[cosOffset], ropeDim_);
        inCosQue_.EnQue(cosLocal);

        local_scalar_t sinLocal = inSinQue_.template AllocTensor<scalar_t>();
        AscendC::DataCopy(sinLocal, sinGm_[cosOffset], ropeDim_);
        inSinQue_.EnQue(sinLocal);

        cosLocal = inCosQue_.template DeQue<scalar_t>();
        sinLocal = inSinQue_.template DeQue<scalar_t>();

        // Get calc buffers as acc_t
        local_acc_t cosAcc = calcBuf_.GetWithOffset<acc_t>(ropeDim_, cosOff_);
        local_acc_t sinAcc = calcBuf_.GetWithOffset<acc_t>(ropeDim_, sinOff_);
        local_acc_t peAcc = calcBuf_.GetWithOffset<acc_t>(ropeDim_, peOff_);
        local_acc_t nopeAcc = calcBuf_.GetWithOffset<acc_t>(kvLoraRank_, nopeOff_);
        local_acc_t tmp0 = calcBuf_.GetWithOffset<acc_t>(ropeDim_, tmp0Off_);
        local_acc_t tmp1 = calcBuf_.GetWithOffset<acc_t>(ropeDim_, tmp1Off_);

        // Cast to float for computation
        AscendC::Cast(cosAcc, cosLocal, AscendC::RoundMode::CAST_NONE, ropeDim_);
        AscendC::Cast(sinAcc, sinLocal, AscendC::RoundMode::CAST_NONE, ropeDim_);
        AscendC::Cast(nopeAcc, kvLocal, AscendC::RoundMode::CAST_NONE, kvLoraRank_);
        AscendC::Cast(peAcc, kvLocal[kvLoraRank_], AscendC::RoundMode::CAST_NONE, ropeDim_);

        // Interleave RoPE on peAcc:
        // For pair (2j, 2j+1): out[2j] = x[2j]*cos[2j] - x[2j+1]*sin[2j]
        //                     out[2j+1] = x[2j]*sin[2j] + x[2j+1]*cos[2j]
        interleave_rope(peAcc, cosAcc, sinAcc, tmp0, tmp1);

        // Allocate output tensors
        local_scalar_t peOut = outPeQue_.template AllocTensor<scalar_t>();
        local_scalar_t nopeOut = outNopeQue_.template AllocTensor<scalar_t>();

        // Cast back to scalar_t
        AscendC::Cast(nopeOut, nopeAcc, AscendC::RoundMode::CAST_TRUNC, kvLoraRank_);
        AscendC::Cast(peOut, peAcc, AscendC::RoundMode::CAST_TRUNC, ropeDim_);

        outPeQue_.EnQue(peOut);
        outNopeQue_.EnQue(nopeOut);

        peOut = outPeQue_.template DeQue<scalar_t>();
        nopeOut = outNopeQue_.template DeQue<scalar_t>();

        // Get slot and write to cache
        int64_t slot = slotsGm_.GetValue(tokenIdx);
        int64_t kCacheOff = slot * ropeDim_;
        int64_t ckvCacheOff = slot * kvLoraRank_;

        AscendC::DataCopy(kCacheGm_[kCacheOff], peOut, ropeDim_);
        AscendC::DataCopy(ckvCacheGm_[ckvCacheOff], nopeOut, kvLoraRank_);

        // Optionally write output tensors
        if (isOutputKv_) {
            AscendC::DataCopy(kPeOutGm_[tokenIdx * ropeDim_], peOut, ropeDim_);
            AscendC::DataCopy(kNopeOutGm_[tokenIdx * kvLoraRank_], nopeOut, kvLoraRank_);
        }

        inKvQue_.FreeTensor(kvLocal);
        inCosQue_.FreeTensor(cosLocal);
        inSinQue_.FreeTensor(sinLocal);
        outPeQue_.FreeTensor(peOut);
        outNopeQue_.FreeTensor(nopeOut);
    }

    __aicore__ inline void compute()
    {
        int coreId = AscendC::GetBlockIdx();
        int numCores = AscendC::GetBlockNum();
        for (int64_t i = coreId; i < numTokens_; i += numCores) {
            process_token(i);
        }
    }

private:
    __aicore__ inline void interleave_rope(
        local_acc_t &pe, const local_acc_t &cos, const local_acc_t &sin,
        local_acc_t &tmp0, local_acc_t &tmp1)
    {
        // pe layout: [x0, x1, x2, x3, ...]
        // We need: out[2j] = x[2j]*cos[2j] - x[2j+1]*sin[2j]
        //          out[2j+1] = x[2j]*sin[2j] + x[2j+1]*cos[2j]
        // Since cos[2j] = cos[2j+1] and sin[2j] = sin[2j+1] in interleave RoPE,
        // we process in pairs using the first element of each pair as cos/sin.
        //
        // tmp0 = pe * cos (element-wise)
        // tmp1 = pe * sin (element-wise)
        // Then: out[2j]   = tmp0[2j] - tmp1[2j+1]  (wait, this doesn't work element-wise)
        //
        // Better approach: use strided DataCopy to separate even/odd, then vector ops
        // For small rope_dim (64), we can do it in a loop

        // Element-pair processing using DataCopy with stride
        // Even elements: pe[0], pe[2], pe[4], ...
        // Odd elements:  pe[1], pe[3], pe[5], ...
        // cos_half: cos[0], cos[2], cos[4], ...
        // sin_half: sin[0], sin[2], sin[4], ...

        // For simplicity and correctness, use scalar ops on float (which IS allowed)
        for (int j = 0; j < halfRopeDim_; ++j) {
            int i0 = 2 * j;
            int i1 = 2 * j + 1;
            float x0 = pe.GetValue(i0);
            float x1 = pe.GetValue(i1);
            float c = cos.GetValue(i0);
            float s = sin.GetValue(i0);

            tmp0.SetValue(i0, x0 * c - x1 * s);
            tmp0.SetValue(i1, x0 * s + x1 * c);
        }
        // Copy result back to pe
        AscendC::Copy(pe, tmp0, ropeDim_, 1, {1, 1, 8, 8});
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inKvQue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inCosQue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inSinQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outPeQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outNopeQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;

    AscendC::GlobalTensor<scalar_t> kvGm_;
    AscendC::GlobalTensor<scalar_t> cosGm_;
    AscendC::GlobalTensor<scalar_t> sinGm_;
    AscendC::GlobalTensor<int64_t> slotsGm_;
    AscendC::GlobalTensor<scalar_t> kCacheGm_;
    AscendC::GlobalTensor<scalar_t> ckvCacheGm_;
    AscendC::GlobalTensor<scalar_t> kPeOutGm_;
    AscendC::GlobalTensor<scalar_t> kNopeOutGm_;

    int kvLoraRank_;
    int ropeDim_;
    int totalDim_;
    int halfRopeDim_;
    int64_t numTokens_;
    bool isOutputKv_;
    int cosOff_, sinOff_, peOff_, nopeOff_, tmp0Off_, tmp1Off_;
};

#define KV_ROPE_CACHE_KERNEL_DECLARE(TYPE)                                                          \
    extern "C" __global__ __aicore__ void kv_rope_cache_##TYPE(                                     \
        __gm__ TYPE* kv, __gm__ TYPE* cos, __gm__ TYPE* sin, __gm__ int64_t* slots,                \
        __gm__ TYPE* k_cache, __gm__ TYPE* ckv_cache,                                              \
        __gm__ TYPE* k_pe_out, __gm__ TYPE* k_nope_out,                                            \
        const int kvLoraRank, const int ropeDim, const int64_t numTokens,                          \
        const int isOutputKv)                                                                      \
    {                                                                                              \
        AscendC::TPipe pipe;                                                                       \
        KvRopeCache<TYPE> op{};                                                                    \
        op.init(kv, cos, sin, slots, k_cache, ckv_cache,                                           \
                k_pe_out, k_nope_out,                                                              \
                kvLoraRank, ropeDim, numTokens,                                                    \
                static_cast<bool>(isOutputKv), &pipe);                                             \
        op.compute();                                                                             \
    }

KV_ROPE_CACHE_KERNEL_DECLARE(half)
#if (__CCE_AICORE__ >= 220)
    KV_ROPE_CACHE_KERNEL_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {

static const int64_t maxParallelSize = 65535;

#define KV_ROPE_CACHE_KERNEL_CALL(TYPE)                                                            \
    kv_rope_cache_##TYPE<<<blockDim, nullptr, stream>>>(                                           \
        reinterpret_cast<TYPE *>(kv), reinterpret_cast<TYPE *>(cos),                              \
        reinterpret_cast<TYPE *>(sin), reinterpret_cast<int64_t *>(slots),                        \
        reinterpret_cast<TYPE *>(k_cache), reinterpret_cast<TYPE *>(ckv_cache),                    \
        reinterpret_cast<TYPE *>(k_pe_out), reinterpret_cast<TYPE *>(k_nope_out),                  \
        kvLoraRank, ropeDim, numTokens, static_cast<int>(isOutputKv));

extern void kv_rope_cache_impl(
    AscendType type, void *stream,
    void *kv, void *cos, void *sin, void *slots,
    void *k_cache, void *ckv_cache,
    void *k_pe_out, void *k_nope_out,
    const int kvLoraRank, const int ropeDim,
    const int64_t numTokens, const bool isOutputKv)
{
    int blockDim = maxParallelSize > numTokens ? numTokens : maxParallelSize;
    if (blockDim < 1) blockDim = 1;

    if (type == AscendType::FP16) {
        KV_ROPE_CACHE_KERNEL_CALL(half);
    }
#if (__CCE_AICORE__ >= 220)
    else if (type == AscendType::BF16) {
        KV_ROPE_CACHE_KERNEL_CALL(bfloat16_t);
    }
#endif
    else {
        return;
    }
}

} // namespace vllm_ascend
