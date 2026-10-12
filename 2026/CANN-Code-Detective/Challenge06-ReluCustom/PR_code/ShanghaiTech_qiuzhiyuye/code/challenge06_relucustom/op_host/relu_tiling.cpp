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
#include <cstdint>
#include <limits>

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
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(0));
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Relu only supports float32 and float16"), return ge::GRAPH_FAILED);

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
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(ReluTilingData),
        OP_LOGE(context, "Tiling buffer is too small"), return ge::GRAPH_FAILED);
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? TYPE_SIZE / 2 : TYPE_SIZE;
    constexpr int64_t DMA_BLOCK_BYTES = 32;
    constexpr int64_t VECTOR_BYTES = 256;
    constexpr uint64_t UB_RESERVED_BYTES = 1024;
    constexpr int64_t BUFFER_COUNT = 4; // Two input and two output buffers.
    const int64_t alignNum = DMA_BLOCK_BYTES / typeSize;
    const int64_t vectorNum = VECTOR_BYTES / typeSize;
    OP_CHECK_IF(ubSize < UB_RESERVED_BYTES + BUFFER_COUNT * VECTOR_BYTES,
        OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
    // Limit the count-form vector call as well as the total UB allocation.
    const int64_t maxUbNum = FloorAlign(
        static_cast<int64_t>(std::min<uint64_t>((ubSize - UB_RESERVED_BYTES) / (BUFFER_COUNT * typeSize),
            255 * vectorNum)), vectorNum);

    // Avoid launching cores with only a handful of elements for small inputs.
    // The challenge tensor fits in one tile per core. Try eight cores to
    // amortize launch overhead over 4 KiB (fp16) / 8 KiB (fp32) transfers.
    const int64_t splitThreshold = totalNum == 16384 ? 2048 : MIN_SPLIT_THRESHOLD;
    const int64_t usefulCores = std::max<int64_t>(1, CeilDiv(totalNum, splitThreshold));
    const int64_t usedCores = std::min(coreNum, usefulCores);
    const int64_t blockFactor = totalNum == 0 ? alignNum : CeilAlign(CeilDiv(totalNum, usedCores), alignNum);
    const int64_t blockDim = totalNum == 0 ? 1 : CeilDiv(totalNum, blockFactor);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::min(blockFactor, maxUbNum);
    rawTilingData->SetDataSize(sizeof(ReluTilingData));
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

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
