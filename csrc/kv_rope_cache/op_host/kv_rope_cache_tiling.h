#ifndef KV_ROPE_CACHE_TILING_H
#define KV_ROPE_CACHE_TILING_H

#include <cstdint>

namespace kv_rope_cache_op {

// Tiling data shared between op_host (CPU) and op_kernel (AIV).
// Compared with CANN's KvRmsNormRopeCacheTilingData, the RMSNorm-only fields
// (reciprocal / epsilon) are dropped since this op does NOT do RMSNorm.
struct KvRopeCacheTilingData {
    int64_t blockSize;      // paged block size (e.g. 128)
    int64_t numHead;         // num_kv_heads
    int64_t seqLength;       // tokens per batch (1 for decode)
    int64_t batchSize;       // batch size
    int64_t numBlocks;       // number of AI-core blocks (== block_dim)
    int64_t blockFactor;     // tokens processed per AI-core block
    int64_t ubFactor;         // tokens processed per UB loop
    int64_t isOutputKv;       // 1: also output k_rope / c_kv (prefill), 0: decode
    int64_t rmsNormLength;   // kv_lora_rank (512)
    int64_t ropeLength;       // qk_rope_head_dim (64)
};

}  // namespace kv_rope_cache

#endif  // KV_ROPE_CACHE_TILING_H
