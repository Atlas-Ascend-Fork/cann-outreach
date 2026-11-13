/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputShape = context->GetInputShape(0);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "Only float16 and float32 are supported"), return ge::GRAPH_FAILED);
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);

    // 每核起点按 16 个元素对齐，两种类型均满足 32 字节对齐。
    int64_t blockDim = totalNum == 0 ? 1 : CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    blockDim = std::min(blockDim, coreNum);
    const int64_t blockFactor = totalNum == 0 ? 16 : CeilAlign(CeilDiv(totalNum, blockDim), int64_t{16});
    blockDim = totalNum == 0 ? 1 : CeilDiv(totalNum, blockFactor);

    // 双缓冲输入/输出队列，加上两块 float32 计算缓冲，预留 8 KB。
    const int64_t elementBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t bytesPerElement = 4 * elementBytes + 2 * sizeof(float);
    OP_CHECK_IF(ubSize <= 8192, OP_LOGE(context, "Insufficient UB"), return ge::GRAPH_FAILED);
    const int64_t maxUbFactor = FloorAlign(
        static_cast<int64_t>((ubSize - 8192) / bytesPerElement), int64_t{64});
    OP_CHECK_IF(maxUbFactor == 0, OP_LOGE(context, "Insufficient UB"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::min(CeilAlign(blockFactor, int64_t{64}), maxUbFactor);
    context->GetRawTilingData()->SetDataSize(sizeof(TanhTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 与 Kernel 中 TILING_KEY_IS(0/1) 的分支对应。
    const uint64_t tilingKey = dtype == ge::DT_FLOAT16
        ? TANH_TPL_SCH_MODE_0
        : TANH_TPL_SCH_MODE_1;
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
