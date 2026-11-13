/*!
 * \file leakyrelu_tiling.cpp
 * \brief Leakyrelu 算子 Tiling 实现
 */

#include <algorithm>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {

constexpr int64_t DATA_BLOCK_BYTES = 32;
constexpr int64_t MIN_ELEMENTS_PER_CORE = 1024;
constexpr int64_t MAX_TILE_ELEMENTS = 2048;

static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetInputShape(0) == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    const float* slope = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(0);
    if (slope == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalNum = 1;
    const auto& shape = context->GetInputShape(0)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 || (dim != 0 && totalNum > std::numeric_limits<int64_t>::max() / dim)) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= dim;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t elementBytes = dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
    const int64_t alignment = DATA_BLOCK_BYTES / elementBytes;
    // 输入、输出各两个队列缓冲；半精度路径另用两个 float 缓冲。
    const int64_t ubBytesPerElement = 4 * elementBytes + (dtype == ge::DT_FLOAT16 ? 2 * sizeof(float) : 0);
    const int64_t tileCapacity = std::min<uint64_t>(ubBytes / ubBytesPerElement, MAX_TILE_ELEMENTS);
    const int64_t alignedCapacity = tileCapacity / alignment * alignment;
    if (coreNum <= 0 || alignedCapacity == 0) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<LeakyreluTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (tiling == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t targetCores = std::min(coreNum, std::max<int64_t>(1, totalNum / MIN_ELEMENTS_PER_CORE));
    const int64_t perCore = totalNum / targetCores + (totalNum % targetCores != 0);
    const int64_t blocks = perCore / alignment + (perCore % alignment != 0);
    if (blocks > std::numeric_limits<int64_t>::max() / alignment) {
        return ge::GRAPH_FAILED;
    }
    const int64_t blockFactor = std::max<int64_t>(1, blocks) * alignment;
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::min(blockFactor, alignedCapacity);
    tiling->negativeSlope = *slope;
    workspace[0] = 0;

    const auto blockDim = totalNum / blockFactor + (totalNum % blockFactor != 0);
    context->SetBlockDim(static_cast<uint32_t>(std::max<int64_t>(1, blockDim)));
    const uint32_t schMode = dtype == ge::DT_FLOAT16 ? LEAKYRELU_TPL_SCH_MODE_0 : LEAKYRELU_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLeakyrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LeakyreluCompileInfo {};

IMPL_OP_OPTILING(Leakyrelu).Tiling(LeakyreluTilingFunc).TilingParse<LeakyreluCompileInfo>(TilingParseForLeakyrelu);

} // namespace optiling
