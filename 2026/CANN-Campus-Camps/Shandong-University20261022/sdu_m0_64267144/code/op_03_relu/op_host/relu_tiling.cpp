/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    if (inputShape == nullptr || inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const auto elementCount = inputShape->GetStorageShape().GetShapeSize();
    if (elementCount <= 0 || elementCount > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalLength = static_cast<uint32_t>(elementCount);
    const uint32_t elementBytes = dtype == ge::DT_FLOAT ? 4U : 2U;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores <= 0) {
        return ge::GRAPH_FAILED;
    }

    // 小张量先采用最多8核，选择可用的最大2次幂核数。
    uint32_t blockDim = 1;
    while (blockDim < 8 &&
           blockDim * 2 <= static_cast<uint32_t>(availableCores)) {
        blockDim *= 2;
    }
    if (totalLength % blockDim != 0) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t TARGET_TILE_LENGTH = 1024;
    const uint32_t blockLength = totalLength / blockDim;

    // tileNum表示每核的缓冲组数，实际循环次数是tileNum * 2。
    uint32_t tileNum = 1;
    while (static_cast<uint64_t>(blockLength) >
           static_cast<uint64_t>(tileNum) * BUFFER_NUM * TARGET_TILE_LENGTH) {
        tileNum *= 2;
    }
    const uint32_t loopCount = tileNum * BUFFER_NUM;
    if (blockLength % loopCount != 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tileLength = blockLength / loopCount;
    if (tileLength == 0 || (tileLength * elementBytes) % 32 != 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // 输入、输出两个队列，各分配两个缓冲区。
    const uint64_t requiredUb =
        2ULL * BUFFER_NUM * tileLength * elementBytes;
    if (requiredUb > ubSize) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = totalLength;
    tiling->tileNum = tileNum;
    context->SetBlockDim(blockDim);

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
