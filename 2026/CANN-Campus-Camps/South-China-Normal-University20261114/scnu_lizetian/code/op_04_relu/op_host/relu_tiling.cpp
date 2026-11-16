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
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// 每核最少处理的数据量（字节）：仅作为兜底下限
constexpr int64_t MIN_BYTES_PER_CORE = 1024;
// 块数硬上限：每块的调度 + 核内初始化开销随块数累加，大 shape 也不开满核
constexpr int64_t MAX_BLOCK_NUM = 24;
// 每核起始地址与单次搬运长度的对齐字节数：512B 对齐可跑满 DMA 带宽（仅 32B 对齐约只有 70%）
constexpr int64_t BLOCK_ALIGN_BYTES = 512;
// 单条 Vector 指令最大 repeat 次数（repeatTimes 为 uint8_t）与单次 repeat 处理的字节数
constexpr int64_t MAX_REPEAT_TIMES = 255;
constexpr int64_t BYTES_PER_REPEAT = 256;
// UB 预留字节数（TPipe/框架自身占用），避免缓冲区占满整个 UB
constexpr int64_t UB_RESERVED_BYTES = 8192;
// 双缓冲块数：与 op_kernel/relu.h 中的 BUFFER_NUM 保持一致
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
// Buffer 份数：输入 x 与输出 y 各一份
constexpr int64_t BUFFER_CNT = 2;
// 单块大小再细分倍数：2 = 单块减半（每核块数翻倍），摊薄流水的填充/排空开销
constexpr int64_t UB_TILE_SPLIT = 2;
// 每个核的目标块数：块数太少流水填充/排空占比高，太多则每块 DMA 启动与同步开销上升
constexpr int64_t TARGET_TILE_NUM = 8;
// 单块最小字节数：小 shape 下延迟主导，DMA 不拆碎
constexpr int64_t MIN_TILE_BYTES = 8192;
// ---------- 核数选取成本模型 ----------
// T(N) = A + c*N + 2*B/(N*BW)：每启动一个 block 约 0.194us，单核 DMA 有效带宽约 30GB/s
constexpr double PER_BLOCK_COST_US = 0.194;
constexpr double DMA_BYTES_PER_US = 30000.0;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
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

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) ? 2 : 4;

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "input shape size is invalid"), return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
    } else {
        // ---------- 1. 多核切分 ----------
        // 按 T(N) = A + c*N + 2B/(N*BW) 的最优解选核数：小 shape 少开核（块启动开销主导），
        // 大 shape 开满核（带宽主导），再用 MAX_BLOCK_NUM 卡住每块固定开销的累加
        int64_t totalBytes = totalNum * typeSize;
        double optCore = std::sqrt(2.0 * static_cast<double>(totalBytes) /
                                   (DMA_BYTES_PER_US * PER_BLOCK_COST_US));
        int64_t usedCoreNum = static_cast<int64_t>(optCore + 0.5);
        usedCoreNum = std::max(usedCoreNum, static_cast<int64_t>(1));
        usedCoreNum = std::min(usedCoreNum, coreNum);
        // 兜底：每核至少 MIN_BYTES_PER_CORE 字节
        int64_t maxCoreByBytes = std::max(CeilDiv(totalBytes, MIN_BYTES_PER_CORE), static_cast<int64_t>(1));
        usedCoreNum = std::min(usedCoreNum, maxCoreByBytes);
        // 块数硬上限
        usedCoreNum = std::min(usedCoreNum, MAX_BLOCK_NUM);

        // 单核处理量按 512B 向上对齐，保证每核 GM 首地址 512B 对齐
        int64_t elemAlign = BLOCK_ALIGN_BYTES / typeSize;
        int64_t blockFactor =
            std::max(CeilDiv(CeilDiv(totalNum, usedCoreNum), elemAlign) * elemAlign, elemAlign);
        // 实际启动的核数（尾核可能不足 blockFactor，由 kernel 侧裁剪）
        int64_t blockNum = CeilDiv(totalNum, blockFactor);

        // ---------- 2. UB 切分 ----------
        // UB 预算 = 双缓冲 × (输入 + 输出) × 细分倍数 共若干块 buffer
        int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
        OP_CHECK_IF(ubBudget <= 0, OP_LOGE(context, "ubSize is too small"), return ge::GRAPH_FAILED);
        int64_t bufferBytes = DOUBLE_BUFFER_NUM * BUFFER_CNT * UB_TILE_SPLIT * typeSize;
        int64_t ubFactorMax = FloorAlign(ubBudget / bufferBytes, elemAlign);
        // 单块需能被一条 Vector 指令覆盖（repeatTimes 上限 255）
        int64_t maxElemPerRepeatGroup = MAX_REPEAT_TIMES * (BYTES_PER_REPEAT / typeSize);
        int64_t ubFactor = std::min(ubFactorMax, maxElemPerRepeatGroup);
        // 按目标块数限制单块大小，保证每核至少 TARGET_TILE_NUM 块，摊薄流水填充/排空开销
        ubFactor = std::min(ubFactor, FloorAlign(CeilDiv(blockFactor, TARGET_TILE_NUM), elemAlign));
        // 单块不得小于 MIN_TILE_BYTES（小 shape 下 DMA 拆碎只会增加启动延迟），但下限不突破 UB 容量
        int64_t minTileElem = std::min(std::max(MIN_TILE_BYTES / typeSize, elemAlign), ubFactorMax);
        ubFactor = std::max(ubFactor, minTileElem);
        // 单块无需超过单核处理量
        ubFactor = std::min(ubFactor, blockFactor);
        // 保证至少有一个对齐单元可算
        ubFactor = std::max(ubFactor, std::min(elemAlign, blockFactor));

        tiling->totalNum = totalNum;
        tiling->blockFactor = blockFactor;
        tiling->ubFactor = ubFactor;

        context->SetBlockDim(static_cast<uint32_t>(blockNum));
    }

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
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
