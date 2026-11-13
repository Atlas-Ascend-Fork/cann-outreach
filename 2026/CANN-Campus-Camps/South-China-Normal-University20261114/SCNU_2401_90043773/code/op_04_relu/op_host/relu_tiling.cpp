/*!
 * \file relu_tiling.cpp
 * \brief ReLU 算子 Tiling 实现
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x_shape = context->GetInputShape(0);

    // 1. 累乘获取总元素数（支持 (8, 2048) 与 (2048, 2048)）
    int64_t total_elem = 1;
    for (size_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        total_elem *= x_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = static_cast<uint32_t>(total_elem);

    // 2. 动态计算 tileNum（以单核大块不超过 16384 元素为基准，防止 UB 越界）
    const uint32_t blockDim = 8;
    uint32_t blockLength = tiling->totalLength / blockDim;
    uint32_t targetTileLen = 16384;
    
    uint32_t calcTileNum = blockLength / targetTileLen;
    if (calcTileNum == 0) {
        calcTileNum = 1;
    }
    // 对齐到双缓冲倍数
    tiling->tileNum = std::max(2u, (calcTileNum / 2) * 2);

    // 3. 配置启动核数
    context->SetBlockDim(blockDim);

    // 4. 配置 workspace 大小
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