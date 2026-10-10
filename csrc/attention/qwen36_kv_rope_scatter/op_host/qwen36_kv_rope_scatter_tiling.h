#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(Qwen36KvRopeScatterTilingData)
TILING_DATA_FIELD_DEF(int64_t, numRow);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, numBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Qwen36KvRopeScatter, Qwen36KvRopeScatterTilingData)

struct Qwen36KvRopeScatterCompileInfo {};

} // namespace optiling
