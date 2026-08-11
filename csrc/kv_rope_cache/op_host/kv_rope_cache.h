#ifndef KV_ROPE_CACHE_OP_HOST_H
#define KV_ROPE_CACHE_OP_HOST_H

#include <tuple>
#include <acl/acl.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "kv_rope_cache_tiling.h"

namespace kv_rope_cache_op {

// MLA fused RoPE + paged-scatter (no RMSNorm), single kernel.
// Computes tiling scalars on the host and returns them (NO GM tiling buffer,
// NO aclrtMemcpy) -> the kernel launch (with scalar args) is fully
// ACL-graph-capturable (synchronous rtMemcpy is rejected in replay mode).
struct KvRopeCacheTiling { int64_t batchSize, seqLength, numHead, blockFactor, ubFactor, numBlocks, isOutputKv, rmsNormLength, ropeLength; uint32_t numBlocks_u; };

inline KvRopeCacheTiling kv_rope_cache_tiling(
    const at::Tensor& kv,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    bool is_output_kv) {
    auto kvShape = kv.sizes();
    TORCH_CHECK(kvShape.size() == 4, "kv must be 4D [B, N, S, D]");
    int64_t batchSize = kvShape[0];
    int64_t numHead = kvShape[1];
    int64_t seqLength = kvShape[2];
    int64_t rmsNormLength = v_cache.size(-1);   // kv_lora_rank (512)
    int64_t ropeLength = k_cache.size(-1);       // qk_rope_head_dim (64)

    int64_t totalTokens = batchSize * seqLength * numHead;
    constexpr int64_t NUM_AIV_CORES = 48;
    uint32_t numBlocks = static_cast<uint32_t>(
        totalTokens < NUM_AIV_CORES ? (totalTokens > 0 ? totalTokens : 1) : NUM_AIV_CORES);
    int64_t blockFactor = (totalTokens + numBlocks - 1) / numBlocks;
    if (blockFactor < 1) blockFactor = 1;
    int64_t ubFactor = blockFactor < 8 ? blockFactor : 8;

    return KvRopeCacheTiling{
        batchSize, seqLength, numHead, blockFactor, ubFactor,
        static_cast<int64_t>(numBlocks), is_output_kv ? 1 : 0,
        rmsNormLength, ropeLength, numBlocks};
}

}  // namespace kv_rope_cache_op

#endif  // KV_ROPE_CACHE_OP_HOST_H
