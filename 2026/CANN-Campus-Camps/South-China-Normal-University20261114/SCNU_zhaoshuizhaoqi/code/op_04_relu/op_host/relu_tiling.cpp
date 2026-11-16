/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 *
 *   1. 每启动一个 block 的固定开销约 0.194us，块越多这笔账越大，
 *      所以大 shape 不是核越多越好，实测封顶 20 块最快。
 *   2. GM 偏移只做 32B 对齐时有效带宽约只有 70%，按 512B 对齐才能跑满，
 *      因此 blockFactor / ubFactor 都按 512B 对齐。
 */

#include <algorithm>
#include <cmath>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;

constexpr int64_t ELEM_BYTES_FP32 = 4;
constexpr int64_t ELEM_BYTES_FP16 = 2;
// 每核最少处理字节数（兜底下限）
constexpr int64_t MIN_BYTES_PER_CORE = 1024;
// 块数硬上限：每块固定开销 ~0.194us，超过这个数就开始亏（实测 20 比 24 更快）
constexpr int64_t MAX_BLOCK_NUM = 20;
// 多核切分粒度：每核 GM 首地址按 512B 对齐
constexpr int64_t BLOCK_ALIGN_BYTES = 512;
// UB 切分粒度：每块起始地址按 512B 对齐
constexpr int64_t UB_ALIGN_BYTES = 512;
// 块数成本模型：T(N) = A + c*N + 2B/(N*BW)
constexpr double PER_BLOCK_COST_US = 0.194;
constexpr double DMA_BYTES_PER_US = 30000.0;
// Vector 单条指令上限
constexpr int64_t MAX_REPEAT_TIMES = 255;
constexpr int64_t BYTES_PER_REPEAT = 256;
// UB 预留
constexpr int64_t UB_RESERVED_BYTES = 8192;
// 与 kernel 侧保持一致：进/出各 2 块
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t BUFFER_CNT = 2;
// 单块再细分倍数：2 -> 每核块数翻倍，流水填充/排空占比下降
constexpr int64_t UB_TILE_SPLIT = 2;
// 目标每核块数：块数 n 时三级流水理论效率 n/(n+1)
constexpr int64_t TARGET_TILE_NUM = 8;
// 单块最小字节数：小 shape 延迟主导，DMA 不要拆碎
constexpr int64_t MIN_TILE_BYTES = 8192;
// 小 shape 快路径：数据量 <=64KB 时延迟主导，固定少核 + 每核 1 块
constexpr int64_t SMALL_TOTAL_BYTES = 65536;
constexpr int64_t SMALL_BLOCK_NUM = 4;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputTotalNum(gert::TilingContext* context, int64_t& totalNum)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t num = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "dynamic shape is not supported"), return ge::GRAPH_FAILED);
        num *= dim;
    }
    totalNum = num;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    int64_t totalNum = 0;
    OP_CHECK_IF(
        GetInputTotalNum(context, totalNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetInputTotalNum error"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    const int64_t dtypeSize = isFp16 ? ELEM_BYTES_FP16 : ELEM_BYTES_FP32;
    context->SetTilingKey(
        isFp16 ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));

    if (totalNum <= 0) {
        tiling->totalNum = totalNum;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t totalBytes = totalNum * dtypeSize;
    const int64_t blockAlignElem = BLOCK_ALIGN_BYTES / dtypeSize;

    // ---------- 1. 小 shape 快路径 ----------
    if (totalBytes <= SMALL_TOTAL_BYTES) {
        int64_t blk = std::min(SMALL_BLOCK_NUM, coreNum);
        blk = std::max(blk, static_cast<int64_t>(1));
        const int64_t blockFactor =
            std::max(CeilDiv(CeilDiv(totalNum, blk), blockAlignElem) * blockAlignElem, blockAlignElem);
        tiling->totalNum = totalNum;
        tiling->blockFactor = blockFactor;
        tiling->ubFactor = blockFactor; // 每核 1 块，无流水填充/排空
        context->SetBlockDim(static_cast<uint32_t>(CeilDiv(totalNum, blockFactor)));
        return ge::GRAPH_SUCCESS;
    }

    // ---------- 2. 选核数 ----------
    const double optCore =
        std::sqrt(2.0 * static_cast<double>(totalBytes) / (DMA_BYTES_PER_US * PER_BLOCK_COST_US));
    int64_t usedCoreNum = static_cast<int64_t>(optCore + 0.5);
    usedCoreNum = std::max(usedCoreNum, static_cast<int64_t>(1));
    usedCoreNum = std::min(usedCoreNum, coreNum);
    const int64_t maxCoreByBytes = std::max(CeilDiv(totalBytes, MIN_BYTES_PER_CORE), static_cast<int64_t>(1));
    usedCoreNum = std::min(usedCoreNum, maxCoreByBytes);
    usedCoreNum = std::min(usedCoreNum, MAX_BLOCK_NUM);

    const int64_t blockFactor =
        std::max(CeilDiv(CeilDiv(totalNum, usedCoreNum), blockAlignElem) * blockAlignElem, blockAlignElem);
    const int64_t blockNum = CeilDiv(totalNum, blockFactor);

    // ---------- 3. 切 UB ----------
    int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
    if (ubBudget <= 0) {
        ubBudget = static_cast<int64_t>(ubSize);
    }
    const int64_t elemAlign = UB_ALIGN_BYTES / dtypeSize;
    const int64_t bufferBytes = DOUBLE_BUFFER_NUM * BUFFER_CNT * UB_TILE_SPLIT * dtypeSize;
    const int64_t ubFactorMax = FloorAlign(ubBudget / bufferBytes, elemAlign);
    int64_t ubFactor = ubFactorMax;

    const int64_t maxElemPerRepeatGroup = MAX_REPEAT_TIMES * (BYTES_PER_REPEAT / dtypeSize);
    ubFactor = std::min(ubFactor, maxElemPerRepeatGroup);
    ubFactor = std::min(ubFactor, FloorAlign(CeilDiv(blockFactor, TARGET_TILE_NUM), elemAlign));

    const int64_t minTileElem = std::min(std::max(MIN_TILE_BYTES / dtypeSize, elemAlign), ubFactorMax);
    ubFactor = std::max(ubFactor, minTileElem);
    ubFactor = std::min(ubFactor, blockFactor);
    ubFactor = std::max(ubFactor, std::min(elemAlign, blockFactor));

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling