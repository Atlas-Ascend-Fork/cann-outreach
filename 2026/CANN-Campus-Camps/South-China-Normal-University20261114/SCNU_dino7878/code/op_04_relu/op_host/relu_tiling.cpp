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

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

namespace {
constexpr uint32_t BLOCK_DIM = 8U;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t BUFFER_NUM_TOTAL = 4;
constexpr uint32_t WS_SYS_SIZE = 0U;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
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
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "invalid dtype"),
        return ge::GRAPH_FAILED);

    uint32_t DT_X = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const gert::StorageShape* inputShapeX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeX);
    int64_t totalIdx = 1;
    for (int64_t i = 0; i < inputShapeX->GetStorageShape().GetDimNum(); i++) {
        totalIdx *= inputShapeX->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(totalIdx <= 0, OP_LOGE(context, "totalIdx is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        totalIdx > static_cast<int64_t>(0xFFFFFFFFU),
        OP_LOGE(context, "totalIdx exceeds uint32"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalLength = static_cast<uint32_t>(totalIdx);

    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
    int64_t alignElems = 32 / typeSize;
    int64_t ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize), typeSize * BUFFER_NUM_TOTAL), alignElems);
    constexpr int64_t MAX_COPY_BYTES = 65504;
    int64_t maxTileByBytes = FloorAlign(MAX_COPY_BYTES / typeSize, alignElems);
    if (ubFactor > maxTileByBytes) {
        ubFactor = maxTileByBytes;
    }
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor is invalid"), return ge::GRAPH_FAILED);

    int64_t blockLength = totalIdx / static_cast<int64_t>(BLOCK_DIM);
    OP_CHECK_IF(blockLength <= 0, OP_LOGE(context, "blockLength is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        totalIdx % static_cast<int64_t>(BLOCK_DIM) != 0,
        OP_LOGE(context, "totalIdx is not divisible by blockDim"),
        return ge::GRAPH_FAILED);

    int64_t needTileNum = CeilDiv(blockLength, ubFactor * BUFFER_NUM);
    int64_t tileNum = 1;
    while (tileNum < needTileNum) {
        tileNum <<= 1;
    }
    int64_t tilePerCore = blockLength / BUFFER_NUM;
    while (tileNum > 1 && tilePerCore % tileNum != 0) {
        tileNum >>= 1;
    }
    OP_CHECK_IF(tileNum <= 0, OP_LOGE(context, "tileNum is invalid"), return ge::GRAPH_FAILED);

    int64_t tileLength = blockLength / tileNum / BUFFER_NUM;
    OP_CHECK_IF(tileLength <= 0, OP_LOGE(context, "tileLength is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tileLength * typeSize * BUFFER_NUM_TOTAL > static_cast<int64_t>(ubSize),
        OP_LOGE(context, "tileLength exceeds UB"),
        return ge::GRAPH_FAILED);

    tiling->tileNum = static_cast<uint32_t>(tileNum);
    context->SetBlockDim(BLOCK_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling