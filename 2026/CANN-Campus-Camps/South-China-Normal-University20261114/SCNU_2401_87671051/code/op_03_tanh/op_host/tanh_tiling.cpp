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

    // 获取输入 dtype：决定 kernel 实例与 32B 对齐的元素数
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const bool isFp16 = (inputDesc->GetDataType() == ge::DT_FLOAT16);
    const int64_t dtypeSize = isFp16 ? 2 : 4;

    // 按 dtype 选择模板实例（half / float）
    uint64_t tilingKey = isFp16 ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
                                : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);

    // 输入 shape 逐维相乘得到总元素数
    const gert::StorageShape* in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    int64_t totalLength = 1;
    for (size_t i = 0; i < in_shape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= in_shape->GetStorageShape().GetDim(i);
    }

    if (totalLength <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // 32B 对齐元素数：fp32 为 8，fp16 为 16
    const int64_t alignElems = 32 / dtypeSize;

    // 评测显示 block 越多初始化开销越大，优先少核一次搬完
    int64_t targetCore = 2;
    int64_t usableCore = coreNum < targetCore ? coreNum : targetCore;
    int64_t blockFactor = (totalLength + usableCore - 1) / usableCore;
    blockFactor = (blockFactor + alignElems - 1) / alignElems * alignElems;
    if (blockFactor <= 0) {
        blockFactor = alignElems;
    }
    int64_t blockDim = (totalLength + blockFactor - 1) / blockFactor;
    if (blockDim < 1) {
        blockDim = 1;
    }

    tiling->totalNum = totalLength;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = blockFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
