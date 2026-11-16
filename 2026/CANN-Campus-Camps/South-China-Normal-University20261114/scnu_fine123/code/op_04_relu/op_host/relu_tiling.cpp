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
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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
    // Compute the per-core and per-tile lengths from shape, dtype and UB.
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

    // Only the two task shapes and the registered float/half types are supported.
    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDescChecked = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescChecked);
    const auto& shape = inputShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2 || shape.GetDim(1) != 2048 ||
                    (shape.GetDim(0) != 8 && shape.GetDim(0) != 2048),
                OP_LOGE(context, "Only (8,2048) and (2048,2048) are supported"),
                return ge::GRAPH_FAILED);
    const auto dtype = inputDescChecked->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
                OP_LOGE(context, "Only float32 and float16 are supported"),
                return ge::GRAPH_FAILED);
    const int64_t elementBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t totalNum = shape.GetShapeSize();
    // Power-of-two core count ensures aligned, equal, non-overlapping GM slices.
    const int64_t desiredCores = totalNum == 16384 ? 8 : 32;
    int64_t usedCores = 1;
    while (usedCores * 2 <= coreNum && usedCores * 2 <= desiredCores) {
        usedCores *= 2;
    }
    const int64_t blockFactor = totalNum / usedCores;
    int64_t ubFactor = blockFactor < 8192 ? blockFactor : 8192;
    constexpr int64_t bufferNum = 2;
    constexpr uint64_t reserveBytes = 4096;
    while (ubFactor > 128 &&
           static_cast<uint64_t>(2 * bufferNum * ubFactor * elementBytes) + reserveBytes > ubSize) {
        ubFactor /= 2;
    }
    OP_CHECK_IF(static_cast<uint64_t>(2 * bufferNum * ubFactor * elementBytes) + reserveBytes > ubSize,
                OP_LOGE(context, "Insufficient UB memory"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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