/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {


constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t SMALL_ROWS = 8;
constexpr int64_t ROW_ELEMENTS = 2048;
constexpr uint64_t MAX_TILE_BYTES = 40U * 1024U;
constexpr uint64_t BUFFER_COUNT = 2;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != ROW_ELEMENTS ||
        (shape.GetDim(0) != SMALL_ROWS && shape.GetDim(0) != 2048)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t rows = shape.GetDim(0);
    const int64_t rowsPerCore = (rows + coreNum - 1) / coreNum;
    const int64_t activeCores = (rows + rowsPerCore - 1) / rowsPerCore;
    const uint64_t typeBytes = dtype == ge::DT_FLOAT16 ? 2U : 4U;
    const uint64_t alignElements = 32U / typeBytes;
    // Each input and output queue owns two UB buffers.
    const uint64_t capacity = ubSize / (2 * BUFFER_COUNT * typeBytes);
    const int64_t alignedCapacity = static_cast<int64_t>(capacity / alignElements * alignElements);
    if (alignedCapacity == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalNum = rows * ROW_ELEMENTS;
    tiling->blockFactor = rowsPerCore * ROW_ELEMENTS;
    // Cap each of four UB buffers at 40 KiB to leave headroom.
    const uint64_t tileByteLimit = rows == SMALL_ROWS ? MAX_TILE_BYTES : 47U * 1024U;
    const int64_t maxTileElements = static_cast<int64_t>(tileByteLimit / typeBytes);
    const int64_t preferredTile =
        tiling->blockFactor < maxTileElements ? tiling->blockFactor : maxTileElements;
    tiling->ubFactor = preferredTile < alignedCapacity ? preferredTile : alignedCapacity;
    context->SetBlockDim(static_cast<uint32_t>(activeCores));
    const bool useStaticSmall = rows == SMALL_ROWS && activeCores == SMALL_ROWS &&
        tiling->blockFactor == ROW_ELEMENTS && tiling->ubFactor == ROW_ELEMENTS;
    if (useStaticSmall) {
        context->SetTilingKey(dtype == ge::DT_FLOAT16
            ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_2)
            : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_3));
    } else {
        context->SetTilingKey(dtype == ge::DT_FLOAT16
            ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
            : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
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
