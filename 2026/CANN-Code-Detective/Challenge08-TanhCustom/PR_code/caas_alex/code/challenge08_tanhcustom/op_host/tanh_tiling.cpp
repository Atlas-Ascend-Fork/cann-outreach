/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BLOCK_DIM = 8;             
constexpr int64_t BUFFER_NUM = 2;            
constexpr int64_t MAX_TILE_BYTES = 32 * 1024;
constexpr int64_t VECTOR_BYTES = 256;

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum <= 0"), return ge::GRAPH_FAILED);

    const auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();

    int64_t typeSize = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = 2;
    } else {
        OP_LOGE(context, "Tanh only supports float32/float16");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const int64_t alignElems = VECTOR_BYTES / typeSize;

    int64_t blockFactor = CeilDiv(totalNum, BLOCK_DIM);
    blockFactor = CeilDiv(blockFactor, alignElems) * alignElems;
    const int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    int64_t ubFactor = std::min(blockFactor, MAX_TILE_BYTES / typeSize);
    if (dtype == ge::DT_FLOAT) {
        ubFactor /= BUFFER_NUM;
    }
    ubFactor = std::max<int64_t>(ubFactor, 1);

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
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
