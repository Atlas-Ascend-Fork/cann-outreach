/*!
 * \file exp_tiling.cpp
 * \brief Exp operator tiling for the Challenge10 input (8, 2048).
 */

#include "register/op_def_registry.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

constexpr uint32_t TILE_NUM = 16;
constexpr uint32_t BLOCK_DIM = 8;

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    auto* tiling = context->GetTilingData<ExpTilingData>();
    const auto& shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalLength *= static_cast<uint32_t>(shape.GetDim(i));
    }
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;
    context->SetBlockDim(BLOCK_DIM);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
