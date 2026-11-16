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

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// ---------------------------------------------------------------------------
// 性能相关常量（小 shape 微秒级算子的核心调优点）
// 本算子典型 shape 为 8*2048 = 16384 个元素（fp32 共 64KB），属于典型的
// "数据量小、核启动头开销占比高" 的场景，因此：
//   1) 核数按数据量自适应：以每核 4096 个元素估算目标核数并封顶到平台 AIV 核数，
//      8*2048 只启动 4 个 AIV，避免几十核并发的固定调度开销；2048*2048 则铺满所有 AIV；
//   2) 每核长度按 512B 对齐，使各核 GM 起始地址落在搬运带宽最优的对齐区间；
//   3) 每核长度 <= 单次处理长度时 Kernel 侧一次搬运 + 一次 Maxs + 一次搬出即可完成，
//      没有循环、没有非对齐尾块（小数据量下尾块 DataCopyPad 很慢）；
//   4) 输入/输出队列各开 2 块（DoubleBuffer），数据量较大需要多 tile 时，
//      下一块的 CopyIn 可与上一块的 Compute/CopyOut 流水重叠。
// ---------------------------------------------------------------------------
constexpr int64_t TARGET_ELEMS_PER_CORE = 4096;  // 每核目标元素数，核数按数据量自适应
constexpr int64_t TARGET_UB_BYTES = 16 * 1024;   // 单块 UB tile 的目标字节数（16KB）
constexpr int64_t GM_ALIGN_BYTES = 512;          // GM 地址对齐粒度，512B 对齐搬运带宽最优
constexpr int64_t UB_ALIGN_BYTES = 32;           // 一个 UB block 的字节数
constexpr int64_t TILING_BUFFER_NUM = 2;         // 与 Kernel 侧 BUFFER_NUM 保持一致
constexpr int64_t UB_RESERVE_BYTES = 8 * 1024;   // UB 预留空间，避免框架/对齐开销导致越界

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

    // 输入 shape 与数据类型
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    // fp16/bf16 走 half 实例，fp32 走 float 实例
    const bool isFp16 = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16);
    const int64_t typeSize = isFp16 ? 2 : TYPE_SIZE;

    // 32B（一个 UB block）是搬运的最小对齐粒度；512B 对齐时 GM 搬运带宽最优
    const int64_t ubAlignNum = UB_ALIGN_BYTES / typeSize;  // fp32=8，fp16=16
    const int64_t gmAlignNum = GM_ALIGN_BYTES / typeSize;  // fp32=128，fp16=256

    tiling->totalNum = totalNum;

    int64_t blockFactor = ubAlignNum;  // 每核处理元素数，至少一个搬运块
    int64_t ubFactor = ubAlignNum;     // 单次 UB 循环处理元素数
    int64_t usedCoreNum = 1;

    if (totalNum > 0) {
        // 1. 目标核数：小 shape 少启核，降低核启动头开销；数据量足够时才多核并行
        int64_t targetCoreNum = 1;
        if (totalNum >= MIN_SPLIT_THRESHOLD) {
            targetCoreNum = CeilDiv(totalNum, TARGET_ELEMS_PER_CORE);
        }
        if (targetCoreNum < 1) {
            targetCoreNum = 1;
        }
        if (targetCoreNum > coreNum) {
            targetCoreNum = coreNum;
        }

        // 2. 每核处理元素数：均分后按 512B 对齐，保证各核 GM 起始地址对齐；
        //    对齐后重新计算真正需要的核数，尾核不越界
        blockFactor = CeilAlign(CeilDiv(totalNum, targetCoreNum), gmAlignNum);
        usedCoreNum = CeilDiv(totalNum, blockFactor);
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        // 3. 单次 UB 循环长度：同时受 UB 容量约束
        //    输入队列 + 输出队列各 TILING_BUFFER_NUM 块（DoubleBuffer）
        int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;
        if (ubBudget < UB_ALIGN_BYTES) {
            ubBudget = UB_ALIGN_BYTES;
        }
        int64_t maxUbElems = FloorAlign(ubBudget / (2 * TILING_BUFFER_NUM * typeSize), ubAlignNum);
        if (maxUbElems < ubAlignNum) {
            maxUbElems = ubAlignNum;
        }

        // 小 shape 下让单核数据一次搬完，避免循环与尾块；大 shape（如 2048*2048）
        // 时按 16KB 一个 tile 切分，保证 DoubleBuffer 流水足够深且 DMA 粒度不过碎
        const int64_t targetUbElems = TARGET_UB_BYTES / typeSize;  // fp32=4096，fp16=8192
        ubFactor = (blockFactor < targetUbElems) ? blockFactor : targetUbElems;
        if (ubFactor > maxUbElems) {
            ubFactor = maxUbElems;
        }
        if (ubFactor > blockFactor) {
            ubFactor = blockFactor;
        }
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey：fp16/bf16 走 half 实例，fp32 走 float 实例
    const uint64_t tilingKey = isFp16 ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
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
