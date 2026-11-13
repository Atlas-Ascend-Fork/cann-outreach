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

const uint32_t TILE_NUM = 8;      // 单核内循环块数
const uint32_t BLOCK_DIM = 8;     // 启动核数

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 根据输入 dtype 选择调度模式 schMode（tiling key）：
    //   float16 -> schMode = 0 -> kernel 实例化 Tanh<half>
    //   float32 -> schMode = 1 -> kernel 实例化 Tanh<float>
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    uint32_t schMode = (DT_X == static_cast<uint32_t>(ge::DT_FLOAT16))
                           ? static_cast<uint32_t>(TANH_TPL_SCH_MODE_0)
                           : static_cast<uint32_t>(TANH_TPL_SCH_MODE_1);
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int64_t totalLength = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= x1_shape->GetStorageShape().GetDim(i);
    }

    // 配置启动核数，并计算每核处理的元素数（host 预计算，核内免除法）
    context->SetBlockDim(BLOCK_DIM);
    uint32_t blockLength = static_cast<uint32_t>(totalLength / BLOCK_DIM);
    uint32_t tileLength = blockLength / TILE_NUM;

    // 填充 tiling 数据
    tiling->blockLength = blockLength; // 每核处理的元素数
    tiling->tileNum = TILE_NUM;        // 单核内循环块数
    tiling->tileLength = tileLength;   // 每块元素数

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
