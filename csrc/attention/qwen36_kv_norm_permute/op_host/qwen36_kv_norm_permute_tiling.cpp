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

#include "qwen36_kv_norm_permute_tiling.h"
#include <algorithm>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_KV_IDX = 0;
constexpr size_t INPUT_GAMMA_IDX = 1;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr int64_t KV_ROW_DIM = 1024;
constexpr int64_t GAMMA_DIM = 256;
} // namespace

ge::graphStatus Tiling4Qwen36KvNormPermute(gert::TilingContext *context)
{
    auto kvDesc = context->GetInputDesc(INPUT_KV_IDX);
    auto gammaDesc = context->GetInputDesc(INPUT_GAMMA_IDX);
    if (kvDesc == nullptr || gammaDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto kvShape = context->GetInputShape(INPUT_KV_IDX)->GetStorageShape();
    auto gammaShape = context->GetInputShape(INPUT_GAMMA_IDX)->GetStorageShape();
    if (kvShape.GetDimNum() != 2 || kvShape.GetDim(1) != KV_ROW_DIM) {
        return ge::GRAPH_FAILED;
    }
    if (gammaShape.GetDimNum() != 1 || gammaShape.GetDim(0) != GAMMA_DIM) {
        return ge::GRAPH_FAILED;
    }
    if (kvDesc->GetDataType() != ge::DT_BF16 || gammaDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    int64_t numRow = kvShape.GetDim(0);

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto epsilonPtr = attrs->GetFloat(0);
    if (epsilonPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float epsilon = *epsilonPtr;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(numRow, coreNum));
    if (blockDim == 0) {
        blockDim = 1;
    }
    context->SetBlockDim(blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    Qwen36KvNormPermuteTilingData tiling;
    tiling.set_numRow(numRow);
    tiling.set_usedCoreNum(blockDim);
    tiling.set_epsilon(epsilon);
    context->SetTilingKey(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4Qwen36KvNormPermute(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Qwen36KvNormPermute)
    .Tiling(Tiling4Qwen36KvNormPermute)
    .TilingParse<Qwen36KvNormPermuteCompileInfo>(TilingPrepare4Qwen36KvNormPermute);

} // namespace optiling
