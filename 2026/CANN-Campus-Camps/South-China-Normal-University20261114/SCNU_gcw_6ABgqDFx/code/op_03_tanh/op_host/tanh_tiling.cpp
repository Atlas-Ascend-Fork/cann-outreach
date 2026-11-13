/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include <algorithm>
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

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

    const auto* shape = context->GetInputShape(0);
    const auto* desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const auto dtype = desc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Unsupported dtype"), return ge::GRAPH_FAILED);
    const int64_t n = shape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(n < 0, OP_LOGE(context, "Unresolved shape"), return ge::GRAPH_FAILED);
    const int64_t bytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t align = 256 / bytes;
    const int64_t cores = std::min(coreNum, std::max<int64_t>(1, n / 1024));
    const int64_t perCore = n / cores + (n % cores != 0);
    const int64_t block = std::max<int64_t>(align, (perCore / align + (perCore % align != 0)) * align);
    // Two input and two output queue slots, plus two float scratch buffers for half input.
    const int64_t scratchBytes = dtype == ge::DT_FLOAT16 ? 2 * sizeof(float) : 0;
    const int64_t perElemBytes = 4 * bytes + scratchBytes;
    const int64_t tile = std::min<int64_t>(
        block, (static_cast<int64_t>(ubSize) / perElemBytes) / align * align);
    OP_CHECK_IF(tile < align, OP_LOGE(context, "Insufficient UB"), return ge::GRAPH_FAILED);
    tiling->totalNum = n;
    tiling->blockFactor = block;
    tiling->ubFactor = tile;
    context->SetBlockDim(n == 0 ? 1 : n / block + (n % block != 0));
    const uint64_t tilingKey = dtype == ge::DT_FLOAT16 ?
        GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
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

