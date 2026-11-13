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
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 2048;
constexpr uint64_t MAX_TILE_BYTES = 16 * 1024;
constexpr uint64_t UB_RESERVED_BYTES = 1024;
constexpr uint64_t BUFFER_NUM = 2;

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
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
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
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Only float32 and float16 are supported"),
        return ge::GRAPH_FAILED);

    const gert::Shape shape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = shape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape"),
        return ge::GRAPH_FAILED);

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignElements = 32 / typeSize;
    // 小输入减少启动核数，大输入使用平台提供的全部 AIV 核。
    int64_t wantedCores = totalNum / MIN_SPLIT_THRESHOLD +
        (totalNum % MIN_SPLIT_THRESHOLD != 0 ? 1 : 0);
    if (wantedCores < 1) {
        wantedCores = 1;
    }
    const int64_t launchCores = wantedCores < coreNum ? wantedCores : coreNum;
    const int64_t averageNum = totalNum / launchCores +
        (totalNum % launchCores != 0 ? 1 : 0);
    int64_t blockFactor = (averageNum / alignElements +
        (averageNum % alignElements != 0 ? 1 : 0)) * alignElements;
    if (blockFactor == 0) {
        blockFactor = alignElements;
    }
    const int64_t blockDim = totalNum == 0 ? 1 :
        totalNum / blockFactor + (totalNum % blockFactor != 0 ? 1 : 0);

    // 输入和输出队列各分配两个 buffer，每块字节数按 32 字节对齐。
    OP_CHECK_IF(ubSize <= UB_RESERVED_BYTES,
        OP_LOGE(context, "Insufficient UB space"), return ge::GRAPH_FAILED);
    uint64_t tileBytes = (ubSize - UB_RESERVED_BYTES) / (2 * BUFFER_NUM);
    if (tileBytes > MAX_TILE_BYTES) {
        tileBytes = MAX_TILE_BYTES;
    }
    tileBytes = tileBytes / 32 * 32;
    OP_CHECK_IF(tileBytes == 0,
        OP_LOGE(context, "Insufficient UB space"), return ge::GRAPH_FAILED);
    int64_t ubFactor = static_cast<int64_t>(tileBytes / typeSize);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
