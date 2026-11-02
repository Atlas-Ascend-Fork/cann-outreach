/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_NUM = 8;
constexpr uint32_t TILE_NUM = 1;

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    if (inputDesc == nullptr || inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    // 本题限定 (8, 2048)。该切分无需尾块处理，两个 dtype 均满足 32 字节搬运对齐。
    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(0) != 8 || shape.GetDim(1) != 2048) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<ExpTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (tiling == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalLength *= static_cast<uint32_t>(shape.GetDim(i));
    }
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    // 用输入 dtype 选择 float / half 模板，避免写死 TilingKey。
    const uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    context->SetBlockDim(BLOCK_NUM);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
