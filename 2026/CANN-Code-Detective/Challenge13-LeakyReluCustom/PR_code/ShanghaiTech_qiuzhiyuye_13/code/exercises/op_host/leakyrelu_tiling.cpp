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
#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    return inShape.GetDimNum() == 0 ? g_vec_1_shape : inShape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
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
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(0));
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Leakyrelu only supports float32 and float16"), return ge::GRAPH_FAILED);

    const auto shape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0 || (dim > 0 && totalNum > std::numeric_limits<int64_t>::max() / dim),
            OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);
        totalNum *= dim;
    }

    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(LeakyreluTilingData),
        OP_LOGE(context, "Tiling buffer is too small"), return ge::GRAPH_FAILED);
    auto* tiling = context->GetTilingData<LeakyreluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* negativeSlope = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, negativeSlope);

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? TYPE_SIZE / 2 : TYPE_SIZE;
    constexpr int64_t DMA_BLOCK_BYTES = 32;
    constexpr int64_t VECTOR_BYTES = 256;
    constexpr uint64_t UB_RESERVED_BYTES = 1024;
    constexpr int64_t BUFFER_COUNT = 4;
    const int64_t alignNum = DMA_BLOCK_BYTES / typeSize;
    const int64_t vectorNum = VECTOR_BYTES / typeSize;
    OP_CHECK_IF(ubSize < UB_RESERVED_BYTES + BUFFER_COUNT * VECTOR_BYTES,
        OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
    const int64_t maxUbNum = FloorAlign(static_cast<int64_t>(std::min<uint64_t>(
        (ubSize - UB_RESERVED_BYTES) / (BUFFER_COUNT * typeSize), 255 * vectorNum)), vectorNum);
    OP_CHECK_IF(maxUbNum <= 0, OP_LOGE(context, "UB tile size is 0"), return ge::GRAPH_FAILED);

    const int64_t splitThreshold = totalNum == 16384 ? 2048 : MIN_SPLIT_THRESHOLD;
    const int64_t usefulCores = std::max<int64_t>(1, CeilDiv(totalNum, splitThreshold));
    const int64_t usedCores = std::min(coreNum, usefulCores);
    const int64_t blockFactor = totalNum == 0 ? alignNum : CeilAlign(CeilDiv(totalNum, usedCores), alignNum);
    const int64_t blockDim = totalNum == 0 ? 1 : CeilDiv(totalNum, blockFactor);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::min(blockFactor, maxUbNum);
    tiling->negativeSlope = *negativeSlope;
    rawTilingData->SetDataSize(sizeof(LeakyreluTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLeakyrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LeakyreluCompileInfo {};
IMPL_OP_OPTILING(Leakyrelu).Tiling(LeakyreluTilingFunc).TilingParse<LeakyreluCompileInfo>(TilingParseForLeakyrelu);

} // namespace optiling
