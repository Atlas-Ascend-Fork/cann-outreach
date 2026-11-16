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

const uint32_t TILE_NUM = 8;
// 与 kernel 侧保持一致：每个队列的 buffer 块数（双缓冲）
const uint32_t BUFFER_NUM = 2;
// 启动的 AIV 核数
constexpr uint32_t BLOCK_DIM = 8;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dtypeX = inputDesc->GetDataType();
    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }

    // 1. 总元素数量
    tiling->totalNum = data_sz;

    // 2. 配置启动核数量：每核数据长度要能被 TILE_NUM * BUFFER_NUM 整除，
    //    且每块字节数满足 32B 对齐（float 8 个元素 / float16 16 个元素为 32B）
    const uint32_t alignElems = (dtypeX == ge::DT_FLOAT16) ? 16U : 8U;
    const uint32_t unit = TILE_NUM * BUFFER_NUM * alignElems;
    uint32_t blockDim = BLOCK_DIM;
    if (blockDim > static_cast<uint32_t>(data_sz) / unit) {
        blockDim = static_cast<uint32_t>(data_sz) / unit;
        if (blockDim == 0) {
            blockDim = 1;
        }
    }
    while (blockDim > 1 && (static_cast<uint32_t>(data_sz) % (blockDim * unit) != 0)) {
        blockDim >>= 1;
    }
    context->SetBlockDim(blockDim);

    // 3. 每核处理的元素数 / 每次 UB 循环处理的元素数（tileNum × DoubleBuffer 分块）
    tiling->blockFactor = data_sz / blockDim;
    tiling->ubFactor = tiling->blockFactor / TILE_NUM / BUFFER_NUM;

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
