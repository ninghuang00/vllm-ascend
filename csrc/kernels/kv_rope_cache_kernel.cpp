#include "kernel_operator.h"
#include <stdio.h>
#include "types.h"
#include "utils.h"

using namespace AscendC;

template <typename scalar_t>
class MinimalKvCopy {
public:
    __aicore__ inline MinimalKvCopy() {}

    __aicore__ inline void init(
        __gm__ scalar_t* kv, __gm__ int64_t* slots, __gm__ scalar_t* k_cache,
        int64_t batchSize, bool isOutputKv, TPipe* pipe)
    {
        pipe_ = pipe;
        batchSize_ = batchSize;
        isOutputKv_ = isOutputKv;
        kvGm_.SetGlobalBuffer(kv);
        kCacheGm_.SetGlobalBuffer(k_cache);
        slotsGm_.SetGlobalBuffer(slots);
        pipe_->InitBuffer(inQue_, 1, 64 * sizeof(scalar_t));
        pipe_->InitBuffer(outQue_, 1, 64 * sizeof(scalar_t));
    }

    __aicore__ inline void compute()
    {
        int64_t coreId = GetBlockIdx();
        int64_t numCores = GetBlockNum();
        constexpr int64_t ROPE_LEN = 64;
        constexpr int64_t RMS_LEN = 512;
        constexpr int64_t D_LEN = RMS_LEN + ROPE_LEN;

        for (int64_t tid = coreId; tid < batchSize_; tid += numCores) {
            // Load k_pe from kv
            LocalTensor<scalar_t> inL = inQue_.AllocTensor<scalar_t>();
            DataCopy(inL, kvGm_[tid * D_LEN + RMS_LEN], ROPE_LEN);
            inQue_.EnQue(inL);
            inL = inQue_.DeQue<scalar_t>();

            // Copy to output
            LocalTensor<scalar_t> outL = outQue_.AllocTensor<scalar_t>();
            DataCopy(outL, inL, ROPE_LEN);
            inQue_.FreeTensor(inL);
            outQue_.EnQue(outL);
            outL = outQue_.DeQue<scalar_t>();

            // Read slot and write to cache
            int64_t slot = slotsGm_.GetValue(tid);
            DataCopy(kCacheGm_[slot * ROPE_LEN], outL, ROPE_LEN);

            outQue_.FreeTensor(outL);
        }
    }

private:
    TPipe* pipe_ = nullptr;
    int64_t batchSize_ = 0;
    bool isOutputKv_ = false;
    GlobalTensor<scalar_t> kvGm_, kCacheGm_;
    GlobalTensor<int64_t> slotsGm_;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
};

#define KV_ROPE_CACHE_PA_KERNEL_DECLARE(TYPE)                                                       \
    extern "C" __global__ __aicore__ void kv_rope_cache_pa_##TYPE(                                    \
        __gm__ TYPE* kv, __gm__ TYPE* cos, __gm__ TYPE* sin, __gm__ int64_t* slots,                 \
        __gm__ TYPE* k_cache, __gm__ TYPE* ckv_cache, __gm__ TYPE* k_pe_out, __gm__ TYPE* k_nope_out, \
        const int64_t rowsPerBlock, const int64_t blockDim, const int64_t batchSize,              \
        const int64_t ubFactor, const int isOutputKv)                                              \
    {                                                                                              \
        TPipe pipe;                                                                               \
        MinimalKvCopy<TYPE> op{};                                                                \
        op.init(kv, slots, k_cache, batchSize, isOutputKv, &pipe);                               \
        op.compute();                                                                            \
    }

KV_ROPE_CACHE_PA_KERNEL_DECLARE(bfloat16_t)
KV_ROPE_CACHE_PA_KERNEL_DECLARE(half)

namespace vllm_ascend {

static const int64_t maxParallelSize = 65535;

#define KV_ROPE_CACHE_PA_KERNEL_CALL(TYPE)                                                          \
    kv_rope_cache_pa_##TYPE<<<blockDim, nullptr, stream>>>(                                        \
        reinterpret_cast<TYPE*>(kv), reinterpret_cast<TYPE*>(cos),                                 \
        reinterpret_cast<TYPE*>(sin), reinterpret_cast<int64_t*>(slots),                          \
        reinterpret_cast<TYPE*>(k_cache), reinterpret_cast<TYPE*>(ckv_cache),                    \
        reinterpret_cast<TYPE*>(k_pe_out), reinterpret_cast<TYPE*>(k_nope_out),                  \
        blockFactor, blockDim, numTokens, ubFactor, static_cast<int>(isOutputKv));

extern void kv_rope_cache_v2_impl(
    AscendType type, void *stream,
    void *kv, void *cos, void *sin, void *slots,
    void *k_cache, void *ckv_cache,
    void *k_pe_out, void *k_nope_out,
    int64_t numTokens, bool isOutputKv)
{
    int64_t blockDim = maxParallelSize > numTokens ? numTokens : maxParallelSize;
    if (blockDim < 1) blockDim = 1;
    int64_t blockFactor = (numTokens + blockDim - 1) / blockDim;
    if (blockFactor < 1) blockFactor = 1;
    int64_t ubFactor = 8;
    if (blockFactor < ubFactor) ubFactor = blockFactor;
    if (ubFactor < 1) ubFactor = 1;

    if (type == AscendType::FP16) {
        KV_ROPE_CACHE_PA_KERNEL_CALL(half);
    }
#if (__CCE_AICORE__ >= 220)
    else if (type == AscendType::BF16) {
        KV_ROPE_CACHE_PA_KERNEL_CALL(bfloat16_t);
    }
#endif
}

} // namespace vllm_ascend
