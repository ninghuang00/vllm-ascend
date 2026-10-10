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

#include "qwen36_kv_rope_scatter_tiling.h"

#include <algorithm>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_KV_IDX = 0;
constexpr size_t INPUT_COS_IDX = 1;
constexpr size_t INPUT_SIN_IDX = 2;
constexpr size_t INPUT_SLOTS_IDX = 3;
constexpr size_t INPUT_CKV_CACHE_IDX = 4;
constexpr size_t INPUT_KPE_CACHE_IDX = 5;
constexpr int64_t KV_ROW_DIM = 576;
constexpr int64_t CKV_DIM = 512;
constexpr int64_t PE_DIM = 64;
} // namespace

ge::graphStatus Tiling4Qwen36KvRopeScatter(gert::TilingContext *context)
{
    auto kvDesc = context->GetInputDesc(INPUT_KV_IDX);
    auto cosDesc = context->GetInputDesc(INPUT_COS_IDX);
    auto sinDesc = context->GetInputDesc(INPUT_SIN_IDX);
    auto slotsDesc = context->GetInputDesc(INPUT_SLOTS_IDX);
    auto ckvCacheDesc = context->GetInputDesc(INPUT_CKV_CACHE_IDX);
    auto kpeCacheDesc = context->GetInputDesc(INPUT_KPE_CACHE_IDX);
    if (kvDesc == nullptr || cosDesc == nullptr || sinDesc == nullptr || slotsDesc == nullptr ||
        ckvCacheDesc == nullptr || kpeCacheDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto kvShape = context->GetInputShape(INPUT_KV_IDX)->GetStorageShape();
    auto cosShape = context->GetInputShape(INPUT_COS_IDX)->GetStorageShape();
    auto sinShape = context->GetInputShape(INPUT_SIN_IDX)->GetStorageShape();
    auto slotsShape = context->GetInputShape(INPUT_SLOTS_IDX)->GetStorageShape();
    auto ckvCacheShape = context->GetInputShape(INPUT_CKV_CACHE_IDX)->GetStorageShape();
    auto kpeCacheShape = context->GetInputShape(INPUT_KPE_CACHE_IDX)->GetStorageShape();

    if (kvShape.GetDimNum() != 2 || kvShape.GetDim(1) != KV_ROW_DIM) {
        return ge::GRAPH_FAILED;
    }
    int64_t numRow = kvShape.GetDim(0);
    if (cosShape.GetDimNum() != 2 || cosShape.GetDim(0) != numRow || cosShape.GetDim(1) != PE_DIM) {
        return ge::GRAPH_FAILED;
    }
    if (sinShape.GetDimNum() != 2 || sinShape.GetDim(0) != numRow || sinShape.GetDim(1) != PE_DIM) {
        return ge::GRAPH_FAILED;
    }
    if (slotsShape.GetDimNum() != 1 || slotsShape.GetDim(0) != numRow) {
        return ge::GRAPH_FAILED;
    }
    if (kvDesc->GetDataType() != ge::DT_BF16 || cosDesc->GetDataType() != ge::DT_BF16 ||
        sinDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    if (slotsDesc->GetDataType() != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }
    if (ckvCacheDesc->GetDataType() != ge::DT_BF16 || kpeCacheDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    // Norm-layout paged caches: [numBlocks, blockSize, 1, dim].
    if (ckvCacheShape.GetDimNum() != 4 || ckvCacheShape.GetDim(2) != 1 || ckvCacheShape.GetDim(3) != CKV_DIM) {
        return ge::GRAPH_FAILED;
    }
    if (kpeCacheShape.GetDimNum() != 4 || kpeCacheShape.GetDim(2) != 1 || kpeCacheShape.GetDim(3) != PE_DIM) {
        return ge::GRAPH_FAILED;
    }
    if (ckvCacheShape.GetDim(0) != kpeCacheShape.GetDim(0) ||
        ckvCacheShape.GetDim(1) != kpeCacheShape.GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    int64_t numBlock = ckvCacheShape.GetDim(0);
    int64_t blockSize = ckvCacheShape.GetDim(1);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(numRow, coreNum));
    if (blockDim == 0) {
        blockDim = 1;
    }
    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    Qwen36KvRopeScatterTilingData tiling;
    tiling.set_numRow(numRow);
    tiling.set_usedCoreNum(blockDim);
    tiling.set_blockSize(blockSize);
    tiling.set_numBlock(numBlock);
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4Qwen36KvRopeScatter(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Qwen36KvRopeScatter)
    .Tiling(Tiling4Qwen36KvRopeScatter)
    .TilingParse<Qwen36KvRopeScatterCompileInfo>(TilingPrepare4Qwen36KvRopeScatter);

} // namespace optiling
