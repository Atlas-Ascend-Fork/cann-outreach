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

const uint32_t BLOCK_DIM = 8;      // 期望启动核数
const uint32_t TILE_ELEMS = 1024;  // 与 kernel 侧保持一致

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，映射为 schMode：fp16 -> 0（half 实例），fp32 -> 1（float 实例）
    // 必须与 op_kernel/tanh.cpp 中的 if constexpr 分支严格对应，不能直接传 dtype 枚举值
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dtype = inputDesc->GetDataType();
    uint32_t schMode = (dtype == ge::DT_FLOAT16) ? TANH_TPL_SCH_MODE_0 : TANH_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int64_t totalNum = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= x1_shape->GetStorageShape().GetDim(i);
    }

    // 填充 tiling：totalNum = 16384，blockFactor = 2048（每核），ubFactor = 1024（每块）
    tiling->totalNum = totalNum;
    tiling->blockFactor = (totalNum + BLOCK_DIM - 1) / BLOCK_DIM;
    tiling->ubFactor = TILE_ELEMS;

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