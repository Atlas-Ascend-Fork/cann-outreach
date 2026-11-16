/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM = 8;
constexpr int64_t UB_FACTOR = 128;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 与 tanh.cpp 中的核入口一致：0 走 half，1 走 float
    const auto* inputDesc = context->GetInputDesc(0);
    uint32_t schMode = inputDesc->GetDataType() == ge::DT_FLOAT16
        ? TANH_TPL_SCH_MODE_0 : TANH_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const auto& shape = x1_shape->GetStorageShape();

    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = totalNum / BLOCK_DIM;
    tiling->ubFactor = UB_FACTOR;
    context->SetBlockDim(BLOCK_DIM);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling