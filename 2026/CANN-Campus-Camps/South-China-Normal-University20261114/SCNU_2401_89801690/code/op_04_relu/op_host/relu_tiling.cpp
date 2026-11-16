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
using Ops::Base::CeilAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t UB_BLOCK_BYTES = 32;
constexpr int64_t MIN_ELEMS_PER_CORE = 2048;
constexpr int64_t PREFERRED_TILE_NUM = 8;
constexpr int64_t UB_USAGE_PERCENT = 90;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0,
                OP_LOGE(context, "coreNum is 0"),
                return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0,
                OP_LOGE(context, "ubSize is 0"),
                return ge::GRAPH_FAILED);

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

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    int64_t totalNum = 1;
    const auto& storageShape = xShape->GetStorageShape();
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        totalNum *= storageShape.GetDim(i);
    }

    const auto dataType = inputDesc->GetDataType();
    const int64_t typeSize =
        (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) ? 2 : 4;
    const int64_t elemsPerBlock = UB_BLOCK_BYTES / typeSize;

    tiling->totalNum = totalNum;

    if (totalNum == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
    } else {
        const int64_t requiredCoreNum =
            CeilDiv(totalNum, MIN_ELEMS_PER_CORE);
        const int64_t usedCoreNum =
            coreNum < requiredCoreNum ? coreNum : requiredCoreNum;

        const int64_t blockFactor =
            CeilAlign(CeilDiv(totalNum, usedCoreNum), elemsPerBlock);

        // DoubleBuffer: 输入 2 份 + 输出 2 份。
        constexpr int64_t DOUBLE_BUFFER_NUM = 2;
        const int64_t usableUbSize =
            static_cast<int64_t>(ubSize) * UB_USAGE_PERCENT / 100;
        const int64_t bytesPerElement =
            typeSize * DOUBLE_BUFFER_NUM * 2;

        const int64_t ubCapacity =
            (usableUbSize / bytesPerElement) / elemsPerBlock * elemsPerBlock;

        OP_CHECK_IF(
            ubCapacity < elemsPerBlock,
            OP_LOGE(context, "UB size is too small for Relu double buffer"),
            return ge::GRAPH_FAILED);

        int64_t ubFactor =
            CeilAlign(CeilDiv(blockFactor, PREFERRED_TILE_NUM),
                      elemsPerBlock);

        if (ubFactor > ubCapacity) {
            ubFactor = ubCapacity;
        }

        tiling->blockFactor = blockFactor;
        tiling->ubFactor = ubFactor;
        context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    }

    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(tilingKey);
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