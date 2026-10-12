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

constexpr int64_t BLOCK_DIM = 8;  // 启动核数
constexpr int64_t TILE_NUM = 16;  // 单核内分块数

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 按 dtype 设置 tilingKey（0: float16 -> half 实例，1: float32 -> float 实例）
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        context->SetTilingKey(0);
    } else {
        context->SetTilingKey(1);
    }

    // 输入 shape 逐维相乘得到总元素数
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= xShape->GetStorageShape().GetDim(i);
    }
    // 多核切分：每核处理 blockFactor 个元素，核内再按 TILE_NUM 分片搬入 UB
    tiling->totalNum = totalNum;
    tiling->blockFactor = totalNum / BLOCK_DIM;
    tiling->ubFactor = tiling->blockFactor / TILE_NUM;

    // 配置启动核数
    context->SetBlockDim(BLOCK_DIM);

    // 配置 workspace 大小
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