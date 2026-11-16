/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
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

// 小 shape：按成本模型选核，并限制在 8 核以内。
// 小 shape 采用“一核一块”，避免把 13KB 左右的数据拆成多个 tile。
constexpr int64_t SMALL_TOTAL_NUM = 32768;
constexpr int64_t MAX_SMALL_CORE_NUM = 8;

// 每个 AIV 核最少处理的字节数。
constexpr int64_t MIN_BYTES_PER_CORE = 1024;

constexpr int64_t MAX_BLOCK_NUM = 20;

// GM 偏移按 512B 对齐时可跑满 DMA 带宽；只做 32B 对齐实测约损失 30%。
constexpr int64_t BLOCK_ALIGN_BYTES = 512;

// 核数成本模型：T(N) = A + c*N + 2*B/(N*BW)
constexpr double PER_BLOCK_COST_US = 0.194;
constexpr double DMA_BYTES_PER_US = 30000.0;

// UB 内 tile 起始地址同时按 512B 对齐。
constexpr int64_t UB_ALIGN_BYTES = 512;

// Vector repeat 上限：repeatTimes 为 uint8_t，单次 repeat 处理 256B。
constexpr int64_t MAX_REPEAT_TIMES = 255;
constexpr int64_t BYTES_PER_REPEAT = 256;

// TPipe/框架自身占用，避免 UB 刚好占满。
constexpr int64_t UB_RESERVED_BYTES = 8192;

// kernel 侧 ping-pong：输入/输出各 DOUBLE_BUFFER_NUM 块。
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t BUFFER_CNT = 2;

constexpr int64_t UB_TILE_SPLIT = 2;
constexpr int64_t TARGET_TILE_NUM = 8;
constexpr int64_t MIN_TILE_BYTES = 8192;

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
        OP_CHECK_IF(
            dim < 0,
            OP_LOGE(context, "dynamic shape is not supported by relu tiling"),
            return ge::GRAPH_FAILED);
        num *= dim;
    }
    totalNum = num;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
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

    int64_t usedCoreNum;
    int64_t blockFactor;
    int64_t blockNum;
    int64_t ubFactor;

    if (totalNum <= SMALL_TOTAL_NUM) {
        const double optCore = std::sqrt(
            2.0 * static_cast<double>(totalBytes) / (DMA_BYTES_PER_US * PER_BLOCK_COST_US));
        usedCoreNum = static_cast<int64_t>(optCore + 0.5);
        usedCoreNum = std::max<int64_t>(usedCoreNum, 1);
        usedCoreNum = std::min<int64_t>(usedCoreNum, MAX_SMALL_CORE_NUM);

        blockFactor = std::max<int64_t>(
            CeilDiv(CeilDiv(totalNum, usedCoreNum), blockAlignElem) * blockAlignElem,
            blockAlignElem);
        blockNum = CeilDiv(totalNum, blockFactor);
        ubFactor = blockFactor;
    } else {
        uint64_t ubSize;
        int64_t coreNum;
        OP_CHECK_IF(
            GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
            OP_LOGE(context, "GetPlatformInfo error"),
            return ge::GRAPH_FAILED);

        const double optCore = std::sqrt(
            2.0 * static_cast<double>(totalBytes) / (DMA_BYTES_PER_US * PER_BLOCK_COST_US));
        usedCoreNum = static_cast<int64_t>(optCore + 0.5);
        usedCoreNum = std::max<int64_t>(usedCoreNum, 1);
        usedCoreNum = std::min<int64_t>(usedCoreNum, coreNum);

        const int64_t maxCoreByBytes =
            std::max<int64_t>(CeilDiv(totalBytes, MIN_BYTES_PER_CORE), 1);
        usedCoreNum = std::min<int64_t>(usedCoreNum, maxCoreByBytes);
        usedCoreNum = std::min<int64_t>(usedCoreNum, MAX_BLOCK_NUM);

        blockFactor = std::max<int64_t>(
            CeilDiv(CeilDiv(totalNum, usedCoreNum), blockAlignElem) * blockAlignElem,
            blockAlignElem);
        blockNum = CeilDiv(totalNum, blockFactor);

        int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
        if (ubBudget <= 0) {
            ubBudget = static_cast<int64_t>(ubSize);
        }

        const int64_t elemAlign = UB_ALIGN_BYTES / dtypeSize;
        const int64_t bufferBytes =
            DOUBLE_BUFFER_NUM * BUFFER_CNT * UB_TILE_SPLIT * dtypeSize;
        const int64_t ubFactorMax = FloorAlign(ubBudget / bufferBytes, elemAlign);
        ubFactor = ubFactorMax;

        const int64_t maxElemPerRepeatGroup =
            MAX_REPEAT_TIMES * (BYTES_PER_REPEAT / dtypeSize);
        ubFactor = std::min<int64_t>(ubFactor, maxElemPerRepeatGroup);
        ubFactor = std::min<int64_t>(
            ubFactor, FloorAlign(CeilDiv(blockFactor, TARGET_TILE_NUM), elemAlign));

        const int64_t minTileElem =
            std::min<int64_t>(std::max<int64_t>(MIN_TILE_BYTES / dtypeSize, elemAlign), ubFactorMax);
        ubFactor = std::max<int64_t>(ubFactor, minTileElem);
        ubFactor = std::min<int64_t>(ubFactor, blockFactor);
        ubFactor = std::max<int64_t>(ubFactor, std::min<int64_t>(elemAlign, blockFactor));
    }

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