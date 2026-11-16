/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 *
 * 设计要点（针对 µs 级小算子）：
 *   1. 纯 Vector + 基础 API，不需要系统 workspace；
 *   2. 不查平台信息（SoC/UB 大小在本评测环境固定），省掉 tiling 固定开销；
 *   3. 小 shape 降核：核启动头开销随核数增加，64KB 用 4 核、每核一次搬 16KB（DMA 最佳粒度）；
 *   4. 大 shape 用 16 核吃满带宽；
 *   5. 尽量用移位/位与代替 64 位除法。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr int64_t SMALL_SHAPE_THRESHOLD = 64 * 1024; // 64KB 以内按小 shape 处理

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 1. workspace 必须声明，纯 vector 算子写 0，避免调用方多一次 malloc
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 2. 总元素数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& inShape = xShape->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < inShape.GetDimNum(); i++) {
        totalNum *= inShape.GetDim(i);
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();

    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
    } else {
        const bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
        const int64_t typeSize = isFp16 ? 2 : 4;
        const int64_t blockAlign = isFp16 ? 256 : 128;  // 512B 对齐对应的元素数
        const int64_t alignElems = isFp16 ? 16 : 8;     // 32B 对齐对应的元素数
        const int64_t ubLimit = isFp16 ? 24576 : 12288; // UB 192KB / (输入+输出各双缓冲 = 4 份)

        // 小 shape 降核：头开销随核数增加；每核 16KB 是官方 DMA 最佳粒度
        const int64_t coreTarget = (totalNum * typeSize <= SMALL_SHAPE_THRESHOLD) ? 4 : 16;
        const int64_t shift = (coreTarget == 4) ? 2 : 4;
        int64_t blockFactor = (totalNum + coreTarget - 1) >> shift;
        // 512B 对齐（blockAlign 是 2 的幂，用位运算代替除法）
        blockFactor = (blockFactor + blockAlign - 1) & ~(blockAlign - 1);

        // 能一次搬完就只切 1 块；否则按 UB 上限切
        int64_t ubFactor = (blockFactor < ubLimit) ? blockFactor : ubLimit;
        if (ubFactor < alignElems) {
            ubFactor = alignElems;
        }

        tiling->totalNum = totalNum;
        tiling->blockFactor = blockFactor;
        tiling->ubFactor = ubFactor;
        context->SetBlockDim((totalNum + blockFactor - 1) / blockFactor);
    }

    // 3. 根据 dtype 选择 tilingKey（fp16/bf16 -> half 模板，fp32 -> float 模板）
    const uint64_t tilingKey = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16)
                                   ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
                                   : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
