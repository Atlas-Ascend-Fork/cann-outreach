/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {
    
const uint32_t TILE_NUM = 16; // 参考 Add 算子，设定单核分块数为 16

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 计算总元素数
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    
    // 设置 tiling 参数
    tiling->totalLength = data_sz;
    tiling->tileNum = TILE_NUM;

    // 配置启动核数（参考 Add 算子，固定为 8 核）
    // 对于 8*2048 = 16384，16384 / 8 = 2048；2048 / 16 = 128，完美 32B 对齐
    context->SetBlockDim(8);

    // 配置 workspace 大小（参考 Add 算子，设为 0）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling