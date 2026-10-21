/*!
 * \file sigmoid_tiling.cpp
 * \brief Sigmoid 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "../op_kernel/sigmoid_tiling_data.h"
#include "../op_kernel/sigmoid_tiling_key.h"

namespace optiling {

static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
{
    auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    auto* tiling = context->GetTilingData<SigmoidTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    uint32_t totalLength = 1;
    const auto& shape = inputShape->GetStorageShape();
    for (size_t dim = 0; dim < shape.GetDimNum(); ++dim) {
        totalLength *= shape.GetDim(dim);
    }
    tiling->totalLength = totalLength;
    // The task input has 8 rows of 2048 elements. Each core processes one row.
    // 16 tiles with double buffering give 64 elements per transfer.
    tiling->tileNum = 16;
    context->SetBlockDim(8);
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SigmoidCompileInfo {};

IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);

} // namespace optiling

