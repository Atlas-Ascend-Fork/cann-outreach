/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

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

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
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

    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Only float32 and float16 are supported"), return ge::GRAPH_FAILED);
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : TYPE_SIZE;
    const int64_t alignNum = 32 / typeSize;

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        OP_CHECK_IF(shape.GetDim(i) < 0,
            OP_LOGE(context, "Input shape must be concrete"), return ge::GRAPH_FAILED);
        totalNum *= shape.GetDim(i);
    }

    // Input and output each have two buffers. Keep all UB tiles 32-byte aligned.
    const int64_t ubCapacity = (static_cast<int64_t>(ubSize) / (4 * typeSize) / alignNum) * alignNum;
    OP_CHECK_IF(ubCapacity == 0,
        OP_LOGE(context, "UB cannot hold one aligned tile"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;
    const int64_t perCore = (totalNum + coreNum - 1) / coreNum;
    tiling->blockFactor = ((perCore + alignNum - 1) / alignNum) * alignNum;
    if (tiling->blockFactor == 0) {
        tiling->blockFactor = alignNum;
    }
    tiling->ubFactor = tiling->blockFactor < ubCapacity ? tiling->blockFactor : ubCapacity;
    const int64_t blockDim = totalNum == 0 ? 1 :
        (totalNum + tiling->blockFactor - 1) / tiling->blockFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ?
        GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};

IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);

} // namespace optiling
