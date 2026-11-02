/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

namespace optiling {


static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetInputShape(0) == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t schMode = dtype == ge::DT_FLOAT ? SQRT_TPL_SCH_MODE_1 : SQRT_TPL_SCH_MODE_0;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    int64_t totalNum = 1;
    const auto& shape = context->GetInputShape(0)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 || (dim != 0 && totalNum > std::numeric_limits<int64_t>::max() / dim)) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= dim;
    }

    constexpr int64_t DATA_BLOCK_BYTES = 32;
    constexpr int64_t MAX_UB_ELEMENTS = 1024;
    constexpr int64_t QUEUE_BUFFER_COUNT = 4;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t elementBytes = dtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const int64_t alignElements = DATA_BLOCK_BYTES / elementBytes;
    if (aivNum == 0 || ubBytes < QUEUE_BUFFER_COUNT * DATA_BLOCK_BYTES) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<SqrtTilingData>();
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (tiling == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalNum = totalNum;
    workspace[0] = 0;
    if (totalNum == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = alignElements;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t average = totalNum / aivNum + (totalNum % aivNum != 0);
    const int64_t blocks = average / alignElements + (average % alignElements != 0);
    if (blocks > std::numeric_limits<int64_t>::max() / alignElements) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockFactor = blocks * alignElements;
    const int64_t ubCapacity = static_cast<int64_t>(std::min<uint64_t>(
        ubBytes / (QUEUE_BUFFER_COUNT * elementBytes), MAX_UB_ELEMENTS));
    const int64_t alignedUb = ubCapacity / alignElements * alignElements;
    tiling->ubFactor = std::min(tiling->blockFactor, alignedUb);
    const uint32_t blockDim = static_cast<uint32_t>(
        totalNum / tiling->blockFactor + (totalNum % tiling->blockFactor != 0));
    context->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};

IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);

} // namespace optiling
