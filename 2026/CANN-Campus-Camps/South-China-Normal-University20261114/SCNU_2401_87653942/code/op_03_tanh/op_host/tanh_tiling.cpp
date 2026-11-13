/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include <algorithm>
#include <limits>

#include "register/op_def_registry.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* platformInfo = context->GetPlatformInfo();
    auto* tiling = context->GetTilingData<TanhTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!inputDesc || !inputShape || !platformInfo || !tiling || !workspace) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const bool isHalf = dtype == ge::DT_FLOAT16;
    const uint32_t elementBytes = isHalf ? 2U : 4U;
    constexpr int64_t dataBlockBytes = 32;
    const int64_t alignElements = dataBlockBytes / elementBytes;

    const auto& shape = inputShape->GetStorageShape();
    const int64_t totalNum = shape.GetDimNum() == 0 ? 1 : shape.GetShapeSize();
    if (totalNum < 0 || totalNum > std::numeric_limits<int64_t>::max() - alignElements + 1) {
        return ge::GRAPH_FAILED;
    }

    workspace[0] = 0;
    context->SetTilingKey(isHalf ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
                                : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1));
    tiling->totalNum = totalNum;
    if (totalNum == 0) {
        tiling->blockFactor = alignElements;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    platform_ascendc::PlatformAscendC platform(platformInfo);
    const int64_t availableCores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (availableCores <= 0 || ubBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    // 以完整的数据搬运块分核，最后一核按实际长度处理尾部。
    const int64_t blocks = 1 + (totalNum - 1) / alignElements;
    const int64_t blocksPerCore = 1 + (blocks - 1) / availableCores;
    const int64_t blockFactor = blocksPerCore * alignElements;
    const int64_t coreCount = 1 + (totalNum - 1) / blockFactor;

    // 两个输入/输出缓冲、两个 FP32 临时缓冲。FP16 再用一个 FP32 转换缓冲。
    const uint64_t ubBytesPerElement = 2 * elementBytes + (isHalf ? 3 : 2) * sizeof(float);
    constexpr uint64_t maxTileBytes = 16 * 1024;
    const auto maxTileElements = std::min(ubBytes / ubBytesPerElement, maxTileBytes / elementBytes);
    const int64_t tileBlocks = std::min(blocksPerCore, static_cast<int64_t>(maxTileElements) / alignElements);
    if (tileBlocks == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = tileBlocks * alignElements;
    context->SetBlockDim(static_cast<uint32_t>(coreCount));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
