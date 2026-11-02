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
    // 获取输入 dtype，按 tilingKey 选择模板实例（float -> MODE_0，float16 -> MODE_1）
    auto* inputDesc = context->GetInputDesc(0);
    uint64_t tilingKey = inputDesc->GetDataType() == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1)
        : GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);

    // 输入 shape 逐维相乘得到总元素数
    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t dataSz = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        dataSz *= static_cast<uint32_t>(xShape->GetStorageShape().GetDim(i));
    }
    tiling->totalLength = dataSz;
    tiling->tileNum = TILE_NUM;

    // 配置启动核数与 tilingKey
    context->SetBlockDim(8);
    context->SetTilingKey(tilingKey);

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
