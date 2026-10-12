/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr int64_t BLOCK_DIM = 32;  // 启动核数（910B3 AIV 数量内尽量吃满并行度）
constexpr int64_t TILE_NUM = 4;    // 单核内分块数

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 按 dtype 设置 tilingKey（0: float16 -> half 实例，1: float32 -> float 实例）
    // tilingKey 必须经 GET_TPL_TILING_KEY 换算，与 ASCENDC_TPL_UINT_SEL 的键值规则一致
    auto inputDesc = context->GetInputDesc(0);
    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    // 输入 shape 逐维相乘得到总元素数（标量 shape 按 1 个元素处理）
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int64_t totalLength = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= xShape->GetStorageShape().GetDim(i);
    }
    // 多核切分：每核处理 blockFactor 个元素，核内按 TILE_NUM 分片搬入 UB（双缓冲）
    tiling->totalNum = totalLength;
    tiling->blockFactor = totalLength / BLOCK_DIM;
    tiling->ubFactor = tiling->blockFactor / TILE_NUM;

    // 配置启动核数
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