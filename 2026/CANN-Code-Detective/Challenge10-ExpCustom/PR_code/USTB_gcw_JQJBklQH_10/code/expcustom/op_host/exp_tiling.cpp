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

const uint32_t TILE_NUM = 16;

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float32 / float16 走不同模板实例）
    uint32_t schMode = EXP_TPL_SCH_MODE_FLOAT32;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        schMode = EXP_TPL_SCH_MODE_FLOAT16;
    }
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 输入 shape 逐维相乘得到总元素数
    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (size_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = data_sz;
    tiling->tileNum = TILE_NUM;

    // 配置启动核数
    context->SetBlockDim(8);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
