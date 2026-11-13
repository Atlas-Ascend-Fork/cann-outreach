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

const uint32_t TILE_NUM = 8;   // 单核内分块数，与 kernel 侧 tileNum 对应
const uint32_t BLOCK_DIM = 8;  // 期望使用的向量核数（多核并行切分）
const uint32_t BUFFER_NUM = 2; // 与 kernel 侧 BUFFER_NUM 一致（双缓冲）

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    if (tiling == nullptr || x1_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& xShape = x1_shape->GetStorageShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        totalLength *= static_cast<uint32_t>(xShape.GetDim(i));
    }

    // 多核切分：每核处理 blockLength = totalLength / blockDim 个元素，
    // kernel 侧每核内再切成 tileNum × BUFFER_NUM 块（每块 tileLength 个元素），
    // 因此要求 totalLength 能被 blockDim × tileNum × BUFFER_NUM 整除；
    // 不满足时逐步折半核数，保证每个核分到的数据都能被完整切块（不丢尾数据）。
    const uint32_t perCoreUnit = TILE_NUM * BUFFER_NUM; // 每核最小元素数
    uint32_t blockDim = BLOCK_DIM;
    while (blockDim > 1 && (totalLength % (blockDim * perCoreUnit) != 0)) {
        blockDim = blockDim >> 1;
    }
    context->SetBlockDim(blockDim);

    // 填写 TilingData 结构体
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;
    // 显式设置 tiling 数据长度，保证 host 侧下发的结构体大小与 kernel 侧解析一致
    context->GetRawTilingData()->SetDataSize(sizeof(TanhTilingData));

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