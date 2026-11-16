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
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    // 考生自行补齐 Tiling实现：逐维相乘求 totalLength
    uint32_t totalLength = 1;
    for (size_t i = 0; i < x1_shape->GetShape().GetDimNum(); ++i) {
        totalLength *= static_cast<uint32_t>(x1_shape->GetShape().GetDim(i));
    }

    // 填充 tiling 数据
    tiling->totalLength = totalLength;
    tiling->tileNum     = TILE_NUM;

    // 设置 AI Core 启动核数（题目输入固定 (8, 2048)，使用 8 核并行，
    // 每核 2048 元素，tileNum = 8，双缓冲后单次 tileLength = 128，满足 32B 对齐）
    constexpr uint32_t BLOCK_DIM = 8;
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