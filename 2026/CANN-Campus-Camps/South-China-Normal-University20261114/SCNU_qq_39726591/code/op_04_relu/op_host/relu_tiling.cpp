/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <cstdint>
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

const uint32_t BLOCK_DIM = 8;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    uint32_t dtype = static_cast<uint32_t>(inputDesc->GetDataType());
    
    // 根据数据类型映射 schMode (0: half, 1: float)
    uint32_t schMode = (dtype == ge::DT_FLOAT) ? RELU_TPL_SCH_MODE_1 : RELU_TPL_SCH_MODE_0;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int64_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    
    tiling->totalNum = data_sz;
    
    // 向上取整，确保非整除 shape 尾数据能被分配到最后一个核
    int64_t blockFactor = (data_sz + BLOCK_DIM - 1) / BLOCK_DIM;
    tiling->blockFactor = blockFactor;

    // 边界保护：若开核数多于数据量，缩减实际使用的核数
    uint32_t usedCores = (data_sz + blockFactor - 1) / blockFactor;
    context->SetBlockDim(usedCores);

    // 块长度按 32B 对齐：fp32 为 8 元素倍数，fp16 为 16 元素倍数
    int64_t alignNum = (dtype == ge::DT_FLOAT) ? 8 : 16;
    int64_t max_ub_factor = 8192;
    int64_t ubFactor = blockFactor < max_ub_factor ? blockFactor : max_ub_factor;
    ubFactor = (ubFactor / alignNum) * alignNum;
    if (ubFactor == 0) {
        ubFactor = alignNum;
    }
    tiling->ubFactor = ubFactor;

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