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

const uint32_t TILE_NUM = 8;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    
    // 补齐 TilingKey 设置，区分 float 和 float16 分支
    if (DT_X == 1) { // fp16
        context->SetTilingKey(GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0));
    } else { // fp32
        context->SetTilingKey(GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1));
    }

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    
    // 计算总元素数
    int32_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
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

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling