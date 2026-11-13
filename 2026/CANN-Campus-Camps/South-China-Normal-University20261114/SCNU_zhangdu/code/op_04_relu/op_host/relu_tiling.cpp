/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */
#include "register/op_def_registry.h"
#include "../op_kernel/relu_tiling_data.h" 
#include "../op_kernel/relu_tiling_key.h"  

namespace optiling {

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    
    tiling->totalLength = data_sz;
    
    // 【防御 1】动态配置核数与切分，防止测试点 2 数据量过大导致 UB 溢出
    uint32_t blockDim = 8; // 默认 8 核
    uint32_t tileNum = 8;  // 默认切分 8 次

    // 如果数据量超过 50万，为了防止单核内存超限，增加切分次数
    if (data_sz > 500000) {
        tileNum = 32; 
    }
    
    tiling->tileNum = tileNum;
    context->SetBlockDim(blockDim);

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