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

    // 动态核数：小形状用少核减少核间分配开销，大形状用多核并行
    uint32_t blockDim = 8;
    if (data_sz <= 16384) {
        blockDim = 2;
    }

    // 910b Unified Buffer 容量 192KB
    constexpr uint32_t UB_BYTES = 192U * 1024U;
    // 输入双缓冲(BUFFER_NUM=2) + 输出单缓冲(1) = 3 块平分 UB
    constexpr uint32_t UB_BLOCKS = 3U;
    // 按数据类型确定单元素字节数（fp16=2B，fp32=4B）
    uint32_t elemSize = (inputDesc->GetDataType() == ge::DT_FLOAT16) ? 2U : 4U;
    // 单块最大字节数 = UB / 块数；最大元素数按此反推
    uint32_t maxTileBytes = UB_BYTES / UB_BLOCKS;
    uint32_t maxTileElems = maxTileBytes / elemSize;
    // 32 字节对齐（Vector 最小搬移单位）
    uint32_t alignElems = 32U / elemSize;
    maxTileElems = (maxTileElems / alignElems) * alignElems;

    uint32_t blockLength = static_cast<uint32_t>(data_sz) / blockDim;
    // 目标单块长度：不超过 UB 上限，也不超过每核数据总量
    uint32_t tileLengthGoal = maxTileElems;
    if (blockLength < tileLengthGoal) {
        tileLengthGoal = blockLength;
    }
    // 向下调整，确保 blockLength 能被 tileLengthGoal 整除（避免尾部余数）
    while (tileLengthGoal > 0U && blockLength % tileLengthGoal != 0U) {
        tileLengthGoal--;
    }
    if (tileLengthGoal == 0U) {
        tileLengthGoal = 1U;
    }
    uint32_t tileNum = blockLength / tileLengthGoal;

    tiling->totalLength = static_cast<uint32_t>(data_sz);
    tiling->tileNum = tileNum;

    // 配置启动核数
    context->SetBlockDim(blockDim);

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
