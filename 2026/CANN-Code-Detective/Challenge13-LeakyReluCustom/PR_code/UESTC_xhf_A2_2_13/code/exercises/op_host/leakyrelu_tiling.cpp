/*!
 * \file leakyrelu_tiling.cpp
 * \brief Leakyrelu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {


constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t UB_ALIGNMENT = 32;
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

static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
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

    LeakyreluTilingData* tiling = context->GetTilingData<LeakyreluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const auto* inputShape = context->GetInputShape(0);
    const auto* desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* slope = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, slope);
    const auto dtype = desc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Only float32 and float16 are supported"), return ge::GRAPH_FAILED);
    const int64_t total = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(total < 0, OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);
    const int64_t bytes = dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
    const int64_t align = UB_ALIGNMENT / bytes;
    // Two buffers per queue plus one temporary vector.
    constexpr uint64_t tileBuffers = 5;
    constexpr uint64_t reserveBytes = 8192;
    OP_CHECK_IF(ubSize < reserveBytes + tileBuffers * UB_ALIGNMENT,
        OP_LOGE(context, "Insufficient UB memory"), return ge::GRAPH_FAILED);
    const int64_t tileCapacity = ((ubSize - reserveBytes) / (tileBuffers * bytes) / align) * align;
    const int64_t wantedCores = total == 0 ? 1 : (total - 1) / MIN_SPLIT_THRESHOLD + 1;
    const int64_t usedCores = wantedCores < coreNum ? wantedCores : coreNum;
    const int64_t perCore = total == 0 ? align : (total - 1) / usedCores + 1;
    const int64_t block = ((perCore - 1) / align + 1) * align;
    const int64_t launchedCores = total == 0 ? 1 : (total - 1) / block + 1;
    tiling->totalNum = total;
    tiling->blockFactor = block;
    tiling->ubFactor = block < tileCapacity ? block : tileCapacity;
    tiling->negativeSlope = *slope;
    tiling->reserved = 0;
    context->SetBlockDim(static_cast<uint32_t>(launchedCores));

    const uint64_t tilingKey = dtype == ge::DT_FLOAT16 ?
        GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLeakyrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LeakyreluCompileInfo {};

IMPL_OP_OPTILING(Leakyrelu).Tiling(LeakyreluTilingFunc).TilingParse<LeakyreluCompileInfo>(TilingParseForLeakyrelu);

} // namespace optiling
