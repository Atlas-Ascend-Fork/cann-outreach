/*!
 * \file relu_tiling.cpp
 * \brief ReLU tiling for the CANNJudge Challenge06 platform template.
 */

#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr int64_t DATA_BLOCK_BYTES = 32;
constexpr int64_t MIN_ELEMENTS_PER_CORE = 1024;
constexpr uint64_t UB_RESERVE_BYTES = 1024;
constexpr uint64_t QUEUE_BUFFER_COUNT = 4; // input/output, two buffers each

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    auto inputShape = context->GetInputShape(0);
    auto tiling = context->GetTilingData<ReluTilingData>();
    auto platformInfo = context->GetPlatformInfo();
    auto workspace = context->GetWorkspaceSizes(1);
    if (inputDesc == nullptr || inputShape == nullptr || tiling == nullptr ||
        platformInfo == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const int64_t elementBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignment = DATA_BLOCK_BYTES / elementBytes;

    int64_t totalNum = 1;
    const auto& shape = inputShape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 || (dim != 0 && totalNum > std::numeric_limits<int64_t>::max() / dim)) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= dim;
    }

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t availableCores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (availableCores <= 0 || ubBytes <= UB_RESERVE_BYTES) {
        return ge::GRAPH_FAILED;
    }
    const int64_t maxTileElements = static_cast<int64_t>(
        (ubBytes - UB_RESERVE_BYTES) / QUEUE_BUFFER_COUNT / DATA_BLOCK_BYTES) * alignment;
    if (maxTileElements == 0) {
        return ge::GRAPH_FAILED;
    }

    // Empty input launches one no-op core; scalar input has one element.
    int64_t blockFactor = alignment;
    int64_t blockCount = 1;
    if (totalNum > 0) {
        const int64_t desiredCores = 1 + (totalNum - 1) / MIN_ELEMENTS_PER_CORE;
        const int64_t usedCores = desiredCores < availableCores ? desiredCores : availableCores;
        const int64_t elementsPerCore = 1 + (totalNum - 1) / usedCores;
        if (elementsPerCore > std::numeric_limits<int64_t>::max() - alignment + 1) {
            return ge::GRAPH_FAILED;
        }
        blockFactor = ((elementsPerCore + alignment - 1) / alignment) * alignment;
        blockCount = 1 + (totalNum - 1) / blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = blockFactor < maxTileElements ? blockFactor : maxTileElements;
    context->SetBlockDim(static_cast<uint32_t>(blockCount));
    workspace[0] = 0;

    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
