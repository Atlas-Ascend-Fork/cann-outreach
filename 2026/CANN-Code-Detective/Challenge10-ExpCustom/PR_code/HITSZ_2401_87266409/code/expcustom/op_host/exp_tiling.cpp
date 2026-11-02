/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MAX_BLOCK_DIM = 8;

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

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
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

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // 总元素数
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        totalNum *= xShape->GetStorageShape().GetDim(i);
    }

    // dtype 决定元素字节数，也决定 tilingKey（half / float 两个实例）
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;

    // 按 8 核切分；最后一个核处理剩余元素
    int64_t blockDim = coreNum < MAX_BLOCK_DIM ? coreNum : MAX_BLOCK_DIM;
    if (blockDim <= 0) {
        blockDim = 1;
    }
    if (totalNum > 0 && blockDim > totalNum) {
        blockDim = totalNum;
    }
    int64_t blockFactor = CeilDiv(totalNum, blockDim);
    if (blockFactor < 1) {
        blockFactor = 1;
    }
    int64_t actualBlockDim = CeilDiv(totalNum, blockFactor);
    if (actualBlockDim < 1) {
        actualBlockDim = 1;
    }

    // UB 双缓冲：输入/输出两个队列 × BUFFER_NUM=2，另留 2 倍安全余量
    int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    int64_t align = ubBlockSize / typeSize;
    if (align < 1) {
        align = 1;
    }
    int64_t ubFactor = static_cast<int64_t>(ubSize) / (8 * typeSize);
    ubFactor = FloorAlign(ubFactor, align);
    if (ubFactor < 1) {
        ubFactor = 1;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(actualBlockDim));

    uint64_t tilingKey = (typeSize == 2) ? GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0)
                                         : GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
