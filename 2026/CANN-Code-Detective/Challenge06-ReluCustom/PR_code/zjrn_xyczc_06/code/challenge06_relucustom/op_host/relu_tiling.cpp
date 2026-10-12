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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t UB_ALIGNMENT = 32;
constexpr int64_t QUEUE_BUFFER_COUNT = 4; // Two input and two output buffers.
constexpr uint64_t UB_RESERVED_BYTES = 1024;

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

    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Only float32 and float16 are supported"), return ge::GRAPH_FAILED);
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignNum = UB_ALIGNMENT / typeSize;
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0 || ubSize <= UB_RESERVED_BYTES,
        OP_LOGE(context, "Invalid shape or UB capacity"), return ge::GRAPH_FAILED);

    int64_t usedCores = totalNum > 0 ? CeilDiv(totalNum, MIN_SPLIT_THRESHOLD) : 1;
    usedCores = usedCores < coreNum ? usedCores : coreNum;
    const int64_t blockFactor = totalNum > 0 ? CeilAlign(CeilDiv(totalNum, usedCores), alignNum) : alignNum;
    usedCores = totalNum > 0 ? CeilDiv(totalNum, blockFactor) : 1;
    // Each element uses four T buffers plus one comparison-mask bit.
    // The reserved space also covers rounding the mask allocation to 32 bytes.
    const int64_t maxUbFactor = FloorAlign(
        static_cast<int64_t>((ubSize - UB_RESERVED_BYTES) * 8 / (QUEUE_BUFFER_COUNT * typeSize * 8 + 1)), alignNum);
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB buffer is too small"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = blockFactor < maxUbFactor ? blockFactor : maxUbFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCores));

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

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
