/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TOTAL_LENGTH = 8 * 2048;
constexpr uint32_t BLOCK_LENGTH = TOTAL_LENGTH / BLOCK_DIM;
constexpr uint32_t TILE_LENGTH = 128;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);

    auto* tiling =
        context->GetTilingData<TanhTilingData>();

    if (inputDesc == nullptr ||
        inputShape == nullptr ||
        tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();

    if (dtype != ge::DT_FLOAT &&
        dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    // 本实现对应题目规定的固定输入形状 (8, 2048)。
    const auto& shape = inputShape->GetStorageShape();

    if (shape.GetDimNum() != 2 ||
        shape.GetDim(0) != 8 ||
        shape.GetDim(1) != 2048) {
        return ge::GRAPH_FAILED;
    }

    // 与你提供的平台核函数入口保持一致：
    // schMode = 0：half
    // schMode = 1：float
    uint32_t schMode = (dtype == ge::DT_FLOAT16)
        ? TANH_TPL_SCH_MODE_0
        : TANH_TPL_SCH_MODE_1;

    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 三个字段的单位都是元素数。
    tiling->totalNum = TOTAL_LENGTH;     // 16384
    tiling->blockFactor = BLOCK_LENGTH;  // 2048
    tiling->ubFactor = TILE_LENGTH;      // 128

    context->SetBlockDim(BLOCK_DIM);

    // 中间结果存放于 UB，不需要额外的 GM workspace。
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh(
    gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh)
    .Tiling(TanhTilingFunc)
    .TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling