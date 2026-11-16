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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t UB_RESERVED_BYTES = 4096;
constexpr int64_t MAX_TILE_BYTES = 32768;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(context, "coreNum is 0"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ubSize);

    OP_CHECK_IF(
        ubSize == 0,
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
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Only float32 and float16 are supported"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const auto& shape = inputShape->GetStorageShape();
    const int64_t totalNum =
        shape.GetDimNum() == 0 ? 1 : shape.GetShapeSize();

    OP_CHECK_IF(
        totalNum < 0,
        OP_LOGE(context, "Invalid input shape"),
        return ge::GRAPH_FAILED);

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignNum = ALIGN_BYTES / typeSize;

    // 小输入避免启动过多核；大输入使用可用的全部 AIV 核。
    int64_t usedCoreNum =
        (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    // 每核处理的元素数量，向上对齐至 32 字节。
    int64_t blockFactor =
        (totalNum + usedCoreNum - 1) / usedCoreNum;

    blockFactor =
        ((blockFactor + alignNum - 1) / alignNum) * alignNum;

    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }

    // 对齐后重新确定实际启动核数。
    usedCoreNum = totalNum == 0
        ? 1
        : (totalNum + blockFactor - 1) / blockFactor;

    OP_CHECK_IF(
        ubSize <= static_cast<uint64_t>(UB_RESERVED_BYTES),
        OP_LOGE(context, "Insufficient UB size"),
        return ge::GRAPH_FAILED);

    // 输入双缓冲 + 输出双缓冲，共四份 tile。
    const int64_t bytesPerElement = 4 * typeSize;
    int64_t ubFactor =
        (static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES)
        / bytesPerElement;

    // 初始性能配置：每份 tile 不超过 32 KiB。
    const int64_t maxTileElements = MAX_TILE_BYTES / typeSize;
    if (ubFactor > maxTileElements) {
        ubFactor = maxTileElements;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    ubFactor = (ubFactor / alignNum) * alignNum;

    OP_CHECK_IF(
        ubFactor < alignNum,
        OP_LOGE(context, "Insufficient UB for one aligned tile"),
        return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 与原模板保持相同的类型分支。
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
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
