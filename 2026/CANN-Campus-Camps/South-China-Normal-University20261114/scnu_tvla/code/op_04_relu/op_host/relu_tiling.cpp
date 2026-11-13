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

// 启动核数（Atlas 训练系列 8 个 AI Core）
constexpr uint32_t CORE_NUM = 8;
// 与 op_kernel/relu.h 中的 BUFFER_NUM 保持一致（双缓冲）
constexpr uint32_t BUFFER_NUM = 2;
// 单块元素数上限：fp32 下 8192 元素 = 32KB，双缓冲下 UB 占用可控在 128KB 以内
constexpr uint32_t MAX_TILE_ELEMS = 8192;
// 单块元素数下限：避免块太小导致循环开销占比过高
constexpr uint32_t MIN_TILE_ELEMS = 32;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }

    // ---- TODO 补齐：设置 tiling 参数 ----
    tiling->totalLength = static_cast<uint32_t>(data_sz);
    uint32_t blockLength = tiling->totalLength / CORE_NUM; // 每核处理元素数
    // 自适应 tileNum：块太大时倍增 tileNum，直到单块元素数不超过上限；
    // 同时保证每块不小于下限、且至少分 2 块，让双缓冲能真正流水起来
    uint32_t tileNum = 1;
    while (blockLength / (tileNum * BUFFER_NUM) > MAX_TILE_ELEMS &&
           blockLength / (tileNum * BUFFER_NUM * 2) >= MIN_TILE_ELEMS) {
        tileNum <<= 1;
    }
    if (tileNum < 2 && blockLength / (2 * BUFFER_NUM) >= MIN_TILE_ELEMS) {
        tileNum = 2; // 至少 2 块，配合 BUFFER_NUM=2 形成流水
    }
    tiling->tileNum = tileNum;

    // ---- TODO 补齐：配置启动核数量 ----
    context->SetBlockDim(CORE_NUM);

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
