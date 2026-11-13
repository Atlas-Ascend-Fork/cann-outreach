/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t UB_BLOCK_BYTES = 32;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t QUEUE_NUM = 2;
constexpr uint64_t UB_RESERVED_BYTES = 8192;
constexpr int64_t MAX_TILE_BYTES = 32768;

static inline int64_t DivUp(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

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
        coreNum <= 0,
        OP_LOGE(context, "coreNum must be positive"),
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

    auto inputDesc = context->GetInputDesc(0);
    auto inputShape = context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const auto dtype = inputDesc->GetDataType();

    OP_CHECK_IF(
        dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Relu supports float32 and float16 only"),
        return ge::GRAPH_FAILED);

    const int64_t typeBytes = (dtype == ge::DT_FLOAT16) ? 2 : 4;
    const int64_t alignNum = UB_BLOCK_BYTES / typeBytes;

    const auto& shape = inputShape->GetStorageShape();
    const int64_t totalNum =
        (shape.GetDimNum() == 0) ? 1 : shape.GetShapeSize();

    OP_CHECK_IF(
        totalNum < 0,
        OP_LOGE(context, "Invalid input shape"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ubSize < UB_RESERVED_BYTES +
                     QUEUE_NUM * BUFFER_NUM * UB_BLOCK_BYTES,
        OP_LOGE(context, "Insufficient UB size"),
        return ge::GRAPH_FAILED);

    // 输入、输出各有两个缓冲区；预留空间后，按 32 字节向下对齐。
    int64_t tileBytes = static_cast<int64_t>(
        (ubSize - UB_RESERVED_BYTES) / (QUEUE_NUM * BUFFER_NUM));

    tileBytes = std::min(tileBytes, MAX_TILE_BYTES);
    tileBytes = tileBytes / UB_BLOCK_BYTES * UB_BLOCK_BYTES;

    int64_t usedCoreNum = 1;
    int64_t blockFactor = alignNum;

    if (totalNum > 0) {
        usedCoreNum = std::min(
            coreNum, DivUp(totalNum, MIN_SPLIT_THRESHOLD));

        blockFactor =
            DivUp(DivUp(totalNum, usedCoreNum), alignNum) * alignNum;

        // 对齐后重新计算，避免启动没有数据的核。
        usedCoreNum = DivUp(totalNum, blockFactor);
    }

    const int64_t ubFactor =
        std::min(blockFactor, tileBytes / typeBytes);

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }

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