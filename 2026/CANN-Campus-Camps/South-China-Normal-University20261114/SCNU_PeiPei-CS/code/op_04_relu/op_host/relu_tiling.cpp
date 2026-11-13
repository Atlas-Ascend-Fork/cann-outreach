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

// 与 op_kernel/relu.h 保持一致：每个队列的 buffer 块数（DoubleBuffer）
constexpr uint32_t BUFFER_NUM = 8;
// GM 与 UB 之间单次搬运的字节对齐粒度
constexpr uint32_t BLOCK_BYTES = 32;
// 单块数据量上限：块太大流水粒度变粗、块太小循环开销变高，32KB 为经验值
constexpr uint32_t MAX_TILE_BYTES = 16 * 1024;
// 拿不到平台信息时的兜底 AIV 核数与 UB 大小（Ascend910B 上的保守值）
constexpr uint32_t DEFAULT_AIV_NUM = 8;
constexpr uint32_t DEFAULT_UB_SIZE = 192 * 1024;
constexpr uint32_t MIN_ELEMS_PER_CORE = 4096;   // 每核最少处理的元素数，用来限制小 shape 的启动核数
constexpr uint32_t GM_ALIGN_BYTES = 4096;   // GM 搬移按 512B 对齐，比 32B 更利于 DMA 的 burst
constexpr uint32_t OUT_BUFFER_NUM = 2;

static inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return a / b + ((a % b) != 0U ? 1U : 0U);
}

static inline uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return (value / align) * align;
}

static inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return CeilDiv(value, align) * align;
}

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

    /* 设置 tiling 参数、配置启动核数量 */
    uint32_t totalLength = static_cast<uint32_t>(data_sz);
    if (totalLength == 0U) {
        OP_LOGE("ReluTiling", "input tensor is empty");
        return ge::GRAPH_FAILED;
    }
    // float32 以 8 元素（32B）为对齐单位，float16 以 16 元素（32B）为对齐单位
    uint32_t elemSize = (inputDesc->GetDataType() == ge::DT_FLOAT16) ? 2U : 4U;
    uint32_t alignElems = GM_ALIGN_BYTES / elemSize;   // 原来是 BLOCK_BYTES / elemSize

    // 平台信息：AIV 核数、UB 大小
    uint32_t aivNum = Ops::Base::GetAivCoreNum(context);
    uint32_t ubSize = Ops::Base::GetUbSize(context);
    if (aivNum == 0U) {
        aivNum = DEFAULT_AIV_NUM;
    }
    if (ubSize == 0U) {
        ubSize = DEFAULT_UB_SIZE;
    }

    // 多核切分：每个核至少处理一个 32B 对齐块，blockFactor 按 32B 向上补齐，
    // 最后一个核不足 blockFactor 的尾巴由 kernel 侧按实际长度处理
    uint32_t maxBlockNum = totalLength / alignElems;
    if (maxBlockNum == 0U) {
        maxBlockNum = 1U;
    }
    if (maxBlockNum > aivNum) {
        maxBlockNum = aivNum;
    }
        uint32_t coreLimit = totalLength / MIN_ELEMS_PER_CORE;
    if (coreLimit < 1U) {
        coreLimit = 1U;
    }
    if (maxBlockNum > coreLimit) {
        maxBlockNum = coreLimit;
    }
    uint32_t blockFactor = AlignUp(CeilDiv(totalLength, maxBlockNum), alignElems);
    uint32_t blockNum = CeilDiv(totalLength, blockFactor);

    // 核内分块：tileLength 不超过 32KB，也不超过 UB 能放下的量（输入、输出两个队列各 BUFFER_NUM 块）
    uint32_t tileBytes = MAX_TILE_BYTES;
    uint32_t ubLimitBytes = ubSize / (BUFFER_NUM + OUT_BUFFER_NUM);
    if (ubLimitBytes < tileBytes) {
        tileBytes = ubLimitBytes;
    }
    uint32_t tileLength = AlignDown(tileBytes / elemSize, alignElems);
    if (tileLength == 0U) {
        tileLength = alignElems;
    }
    uint32_t maxTileLength = AlignDown(blockFactor, alignElems);
    if (maxTileLength >= alignElems && tileLength > maxTileLength) {
        tileLength = maxTileLength;
    }

    /* 写入平台模板的 ReluTilingData：{totalNum, blockFactor, ubFactor}
     *   totalNum    总元素数
     *   blockFactor 每核处理元素数（kernel 用它算 GM 偏移与单核数据量）
     *   ubFactor    一轮双缓冲（BUFFER_NUM 块）搬运的元素数，kernel 中 tileLength = ubFactor / BUFFER_NUM
     */
    tiling->totalNum = totalLength;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = tileLength * BUFFER_NUM;

    // 配置启动核数
    context->SetBlockDim(blockNum);

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
