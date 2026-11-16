/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

static constexpr uint32_t BUFFER_NUM = 2;  // kernel 侧每队列 buffer 块数（双缓冲）
static constexpr uint32_t BLOCK_SIZE = 32; // 32B 对齐

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey：
    // float16 -> MODE_0（half 实例），float32 -> MODE_1（float 实例）
    auto inputDesc = context->GetInputDesc(0);
    uint32_t schMode = (inputDesc->GetDataType() == ge::DT_FLOAT16) ?
                       RELU_TPL_SCH_MODE_0 : RELU_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 输入 shape 逐维相乘得到总元素数
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int64_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    /* TODO 考生自行补齐
    设置tiling参数
    配置启动核数量
    */
    if (data_sz <= 0) {
        return ge::GRAPH_FAILED;
    }

    // 获取平台信息：AIV 核数、UB 大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    // 配置启动核数量：不超过元素个数，避免空核
    uint32_t usedCoreNum = coreNum;
    if (data_sz < static_cast<int64_t>(usedCoreNum)) {
        usedCoreNum = static_cast<uint32_t>(data_sz);
    }
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }

    // 每个核处理的元素数（向上取整；kernel 中尾核会截断）
    int64_t blockFactor = (data_sz + usedCoreNum - 1) / usedCoreNum;

    // 每次 UB 循环处理的元素数：
    // UB 同时容纳输入、输出两个队列，每个队列 BUFFER_NUM 块缓冲
    const uint32_t dtypeSize = (inputDesc->GetDataType() == ge::DT_FLOAT16) ? 2 : 4;
    const uint32_t alignElements = BLOCK_SIZE / dtypeSize;
    int64_t maxUbFactor = static_cast<int64_t>(ubSize / (dtypeSize * 2 * BUFFER_NUM));
    maxUbFactor = maxUbFactor / alignElements * alignElements;
    if (maxUbFactor < static_cast<int64_t>(alignElements)) {
        maxUbFactor = alignElements;
    }

    // 尽量让每次搬运覆盖整个核的数据；数据量大时用满 UB
    int64_t ubFactor = (blockFactor < maxUbFactor) ? blockFactor : maxUbFactor;
    ubFactor = (ubFactor + alignElements - 1) / alignElements * alignElements;
    if (ubFactor > maxUbFactor) {
        ubFactor = maxUbFactor;
    }

    // 设置 tiling 参数
    tiling->totalNum = data_sz;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 配置启动核数量
    context->SetBlockDim(usedCoreNum);

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
