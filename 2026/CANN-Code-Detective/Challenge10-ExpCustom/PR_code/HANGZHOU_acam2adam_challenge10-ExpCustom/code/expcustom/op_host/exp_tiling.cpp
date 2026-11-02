/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

namespace {
// 每核最小数据量：32768 bit = 4KB（按元素位宽折算）
constexpr int32_t MIN_TILING_BITS_SIZE_PER_CORE = 32768;
// 元素对齐因子：512 元素（保证 32B/256B 对齐）
constexpr int32_t ELEM_ALIGN_FACTOR = 512;
// 256B 对齐
constexpr int32_t ALIGN_256 = 256;
// UB 队列数：in/out 队列 × 每队列 BUFFER_NUM(=2)
constexpr int32_t BUFFER_QUEUE_NUM = 4;
// ExpTilingData 全部字段为 int32，可表示的正区间上限（R7：越界 fail-fast，禁止静默截断）
constexpr int64_t TILING_FIELD_MAX = static_cast<int64_t>(std::numeric_limits<int32_t>::max());

// 展平输入 shape 计算元素总数：int64 累乘并做溢出检查；
// 遇到负维度返回 false（非法 shape），遇到 0 维直接判为空 tensor（totalNum=0）。
bool CalcTotalNum(const gert::Shape& storageShape, int64_t& totalNum)
{
    totalNum = 1;
    for (size_t i = 0; i < storageShape.GetDimNum(); i++) {
        const int64_t dim = storageShape.GetDim(i);
        if (dim < 0) {
            return false; // 非法负维度
        }
        if (dim == 0) {
            totalNum = 0; // 空 tensor：后续维度不参与累乘
            return true;
        }
        if (totalNum > std::numeric_limits<int64_t>::max() / dim) {
            return false; // int64 累乘溢出
        }
        totalNum *= dim;
    }
    return true;
}

} // namespace

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // dtype 白名单（R5）：仅支持 FLOAT/FLOAT16，其余显式拒绝；
    // elemBytes 与 tilingKey 的选择只在该白名单内分支，禁止按 2 字节兜底。
    // tilingKey 编码：float32 -> 0（C_DT_FLOAT），float16 -> 1（C_DT_FLOAT16）。
    const ge::DataType dtype = inputDesc->GetDataType();
    int32_t elemBytes = 0;
    if (dtype == ge::DT_FLOAT) {
        elemBytes = 4;
        ASCENDC_TPL_SEL_PARAM(context, C_DT_FLOAT);
    } else if (dtype == ge::DT_FLOAT16) {
        elemBytes = 2;
        ASCENDC_TPL_SEL_PARAM(context, C_DT_FLOAT16);
    } else {
        return ge::GRAPH_FAILED;
    }

    auto tiling = context->GetTilingData<ExpTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 元素总数（R7）：int64 累乘（带溢出检查）；ExpTilingData 字段为 int32，
    // 超出 int32 正区间即 fail-fast（> 2^31-1 元素不支持），禁止静默截断。
    int64_t totalNum = 0;
    if (!CalcTotalNum(inputShape->GetStorageShape(), totalNum)) {
        return ge::GRAPH_FAILED;
    }
    if (totalNum > TILING_FIELD_MAX) {
        return ge::GRAPH_FAILED;
    }

    // 空输入（totalNum=0）：按赛道/题面口径用 SetBlockDim 显式下发最小合法 block dim（1），
    // 并清零 9 个 tiling 字段；kernel 侧 blockNum<=0 早退，确保 0 元素不读写 GM。
    if (totalNum <= 0) {
        if (context->SetBlockDim(1U) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        auto ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) {
            ws[0] = 0; // 本方案无 workspace
        }
        tiling->totalNum = 0;
        tiling->coreNum = 0;
        tiling->blockFactor = 0;
        tiling->blockNum = 0;
        tiling->ubFactor = 0;
        tiling->ubLoopOfFormerBlock = 0;
        tiling->ubTailOfFormerBlock = 0;
        tiling->ubLoopOfTailBlock = 0;
        tiling->ubTailOfTailBlock = 0;
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t availCoreNum = Ops::Base::GetAivCoreNum(context);
    const uint32_t ubSize = Ops::Base::GetUbSize(context);
    if (availCoreNum == 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    // 多核切分：每核至少 4KB 数据（512 元素对齐）
    const int32_t minDtypeBits = elemBytes * 8;
    int64_t coreNum64 =
        (totalNum * minDtypeBits + MIN_TILING_BITS_SIZE_PER_CORE - 1) / MIN_TILING_BITS_SIZE_PER_CORE;
    coreNum64 = coreNum64 < static_cast<int64_t>(availCoreNum) ? coreNum64
                                                               : static_cast<int64_t>(availCoreNum);
    if (coreNum64 < 1) {
        coreNum64 = 1;
    }
    const int32_t coreNum = static_cast<int32_t>(coreNum64);

    const int64_t blockFactor64 = ((totalNum + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) /
                                  ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    const int64_t blockNum64 = (totalNum + blockFactor64 - 1) / blockFactor64;
    // R7：blockFactor / blockNum 转 int32 前校验正区间，越界 fail-fast（禁止静默截断）
    if (blockFactor64 <= 0 || blockFactor64 > TILING_FIELD_MAX ||
        blockNum64 <= 0 || blockNum64 > TILING_FIELD_MAX) {
        return ge::GRAPH_FAILED;
    }
    const int32_t blockNum = static_cast<int32_t>(blockNum64);
    const int32_t blockFactor = static_cast<int32_t>(blockFactor64);

    // UB tile：256B 对齐；4 块队列均分 UB 容量
    const int32_t maxElemNum = static_cast<int32_t>(ubSize / (BUFFER_QUEUE_NUM * elemBytes));
    int32_t ubFactor = maxElemNum / (ALIGN_256 / elemBytes) * (ALIGN_256 / elemBytes);
    ubFactor = ubFactor < blockFactor ? ubFactor : blockFactor;
    if (ubFactor <= 0) {
        return ge::GRAPH_FAILED; // UB 容量不足以构造合法 tile（同时避免后续除零）
    }

    // loop/tail 中间量用 int64 计算，避免 blockFactor + ubFactor 的 int32 相加溢出；
    // 结果天然落在 [1, blockFactor] ⊆ int32 正区间内。
    const int32_t ubLoopOfFormerBlock =
        static_cast<int32_t>((static_cast<int64_t>(blockFactor) + ubFactor - 1) / ubFactor);
    const int32_t ubTailOfFormerBlock = static_cast<int32_t>(
        static_cast<int64_t>(blockFactor) - static_cast<int64_t>(ubLoopOfFormerBlock - 1) * ubFactor);

    const int64_t blockTail = totalNum - static_cast<int64_t>(blockNum - 1) * blockFactor;
    const int32_t ubLoopOfTailBlock = static_cast<int32_t>((blockTail + ubFactor - 1) / ubFactor);
    const int32_t ubTailOfTailBlock =
        static_cast<int32_t>(blockTail - static_cast<int64_t>(ubLoopOfTailBlock - 1) * ubFactor);

    // block dim：每个 block 恰好对应一个核，实际启动 blockNum 个核。
    // R3：按赛道/题面口径使用 SetBlockDim 下发启动核数；
    // 必须校验返回值，失败时返回 GRAPH_FAILED，禁止静默使用框架默认核数。
    if (context->SetBlockDim(static_cast<uint32_t>(blockNum)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) {
        ws[0] = 0; // 本方案无 workspace
    }

    tiling->totalNum = static_cast<int32_t>(totalNum);
    tiling->coreNum = coreNum;
    tiling->blockFactor = blockFactor;
    tiling->blockNum = blockNum;
    tiling->ubFactor = ubFactor;
    tiling->ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    tiling->ubTailOfFormerBlock = ubTailOfFormerBlock;
    tiling->ubLoopOfTailBlock = ubLoopOfTailBlock;
    tiling->ubTailOfTailBlock = ubTailOfTailBlock;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
