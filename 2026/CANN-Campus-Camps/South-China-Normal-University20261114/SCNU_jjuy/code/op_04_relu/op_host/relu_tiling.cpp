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

#include <algorithm>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t UB_BLOCK_BYTES = 32;
constexpr uint64_t UB_RESERVED_BYTES = 8192;
constexpr int64_t MAX_TILE_BYTES = 32768;
constexpr int64_t QUEUE_BUFFER_COUNT = 4;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

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

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "Relu only supports float16 and float32"),
        return ge::GRAPH_FAILED);

    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : TYPE_SIZE;
    const int64_t alignNum = UB_BLOCK_BYTES / typeSize;

    // 小张量避免启动过多核，大张量使用全部可用的 Vector 核。
    int64_t blockDim = std::min(coreNum, std::max(int64_t{1}, CeilDiv(totalNum, MIN_SPLIT_THRESHOLD)));
    const int64_t blockFactor = std::max(alignNum, CeilAlign(CeilDiv(totalNum, blockDim), alignNum));
    blockDim = std::max(int64_t{1}, CeilDiv(totalNum, blockFactor));

    // 输入、输出分别使用两个缓冲，四份 tile 必须同时放入 UB。
    OP_CHECK_IF(
        ubSize <= UB_RESERVED_BYTES,
        OP_LOGE(context, "Insufficient UB size"),
        return ge::GRAPH_FAILED);
    int64_t tileBytes = static_cast<int64_t>((ubSize - UB_RESERVED_BYTES) / QUEUE_BUFFER_COUNT);
    tileBytes = std::min(tileBytes, MAX_TILE_BYTES);
    tileBytes = tileBytes / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
    OP_CHECK_IF(tileBytes == 0, OP_LOGE(context, "Insufficient UB for one tile"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::min(blockFactor, tileBytes / typeSize);
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    const uint64_t tilingKey = dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
