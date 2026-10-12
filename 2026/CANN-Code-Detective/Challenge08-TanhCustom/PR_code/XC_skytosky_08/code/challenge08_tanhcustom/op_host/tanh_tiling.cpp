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


const uint32_t TILE_NUM = 1; // 每核单块一次搬入/计算/搬出，避免多次小循环的 DMA 串行开销
const uint32_t BLOCK_NUM = 8;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE("Tanh", "input desc is nullptr");
        return ge::GRAPH_FAILED;
    }
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 判空
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    if (x1_shape == nullptr) {
        OP_LOGE("Tanh", "input shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    // 输入 shape 逐维相乘得到总元素数（64 位累乘防溢出，超限显式报错而非静默回绕）
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    int64_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    if (data_sz > UINT32_MAX) {
        OP_LOGE("Tanh", "total element num %ld exceeds uint32 range", data_sz);
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = static_cast<uint32_t>(data_sz);
    tiling->tileNum = TILE_NUM;

    // 配置启动核数
    context->SetBlockDim(BLOCK_NUM);

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
