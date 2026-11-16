/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

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
    // ================================================================
    // 阶段1：获取平台信息（UB 大小、AIV 核数）
    // ================================================================
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // ================================================================
    // 阶段2：获取输入 shape 与 dtype
    // ================================================================
    // 逐维相乘得到总元素数（标量 shape 的 dimNum 为 0，视为 1 个元素）
    const gert::StorageShape* inputShapeX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeX);
    int64_t totalIdx = 1;
    for (int64_t i = 0; i < inputShapeX->GetStorageShape().GetDimNum(); i++) {
        totalIdx *= inputShapeX->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(totalIdx <= 0, OP_LOGE(context, "totalIdx is invalid"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "invalid dtype"),
        return ge::GRAPH_FAILED);
    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;

    // ================================================================
    // 阶段3：设置 Workspace 大小
    // ================================================================
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // ================================================================
    // 阶段4：计算 tiling 数据并下发到 kernel 侧
    // ================================================================
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 一次处理 128 个元素。按整块分核，避免相邻核写同一段 GM。
    constexpr int64_t tile = 128;
    const int64_t tiles = CeilDiv(totalIdx, tile);
    const int64_t maxCores = coreNum < 8 ? coreNum : 8;
    const int64_t block = CeilDiv(tiles, maxCores) * tile;
    const int64_t usedCores = CeilDiv(totalIdx, block);
    // 输入输出队列各两块、五个 FP32 工作区和一个 32 字节比较掩码。
    const uint64_t bytesNeeded = tile * (4 * typeSize + 5 * sizeof(float)) + 32;
    OP_CHECK_IF(ubSize < bytesNeeded, OP_LOGE(context, "UB too small"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = block;
    tiling->ubFactor = tile;
    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
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