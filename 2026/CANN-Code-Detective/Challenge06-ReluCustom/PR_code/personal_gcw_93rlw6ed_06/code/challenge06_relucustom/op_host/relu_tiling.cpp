/*!
 * ile relu_tiling.cpp
 * rief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

const uint32_t TILE_NUM = 16;
const uint32_t BLOCK_DIM = 8;
// 单核 UB 容量约 192KB，float32 最坏情况 4 字节，input+output 各一份
const uint32_t MAX_UB_FACTOR = 192 * 1024 / 4 / 2; // = 24576

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    uint32_t schMode = inputDesc->GetDataType() == ge::DT_FLOAT16 ? RELU_TPL_SCH_MODE_0 : RELU_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    int64_t totalNum = 1;
    for (int64_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= x_shape->GetStorageShape().GetDim(i);
    }
    if (totalNum <= 0) {
        OP_LOGE(context, "invalid totalNum: %ld", totalNum);
        return ge::GRAPH_FAILED;
    }

    // 多核切分：每核标准处理量向上取整，保证 8*blockFactor >= totalNum 覆盖全部元素
    int64_t blockFactor = (totalNum + BLOCK_DIM - 1) / BLOCK_DIM;
    // 小张量（totalNum < blockDim）时按实际元素数缩减核数，避免空核越界
    int64_t usedCoreNum = BLOCK_DIM;
    if (totalNum < BLOCK_DIM) {
        usedCoreNum = totalNum;
        blockFactor = 1;
    }
    // 尾核实际元素数（前 N-1 核均为 blockFactor，尾核处理剩余）
    int64_t lastBlockLength = totalNum - blockFactor * (usedCoreNum - 1);
    if (lastBlockLength <= 0) {
        lastBlockLength = blockFactor; // 防御：正常不会发生
    }

    // 核内分块：ubFactor 取 blockFactor/TILE_NUM 与 UB 容量上限的较小值，保证非零
    int64_t ubFactor = blockFactor / TILE_NUM;
    if (ubFactor > MAX_UB_FACTOR) {
        ubFactor = MAX_UB_FACTOR;
    }
    if (ubFactor == 0) {
        ubFactor = blockFactor; // 小张量退化为单块
    }

    tiling->totalNum = static_cast<uint32_t>(totalNum);
    tiling->blockFactor = static_cast<uint32_t>(blockFactor);
    tiling->lastBlockLength = static_cast<uint32_t>(lastBlockLength);
    tiling->ubFactor = static_cast<uint32_t>(ubFactor);

    // 配置启动核数（小张量时用实际核数）
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

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
