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
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t GM_ALIGN_BYTES = 512;        // 512B 对齐
constexpr int64_t FAST_TOTAL_0 = 16384;
constexpr int64_t FAST_TOTAL_1 = 92160;
constexpr int64_t FAST_CORE_0 = 16;            // 小 shape 摊薄每核数据，掩盖启动开销
constexpr int64_t FAST_CORE_1 = 8;
constexpr int64_t GENERIC_BLOCK_TARGET = 8192;
constexpr int64_t FALLBACK_MAX_UB_FACTOR = 32768;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto platform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "invalid AIV core number"), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "invalid UB size"), return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "unsupported dtype"),
        return ge::GRAPH_FAILED);

    const int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
    const int64_t gmAlignNum = GM_ALIGN_BYTES / typeSize;

    int64_t usedCoreNum = 1;
    int64_t blockFactor = totalNum;
    int64_t ubFactor = totalNum;

    if (totalNum == FAST_TOTAL_0 && coreNum >= FAST_CORE_0) {
        // 16384 的最优路径：4 核，一次搬运完成，减少 block 启动开销。
        usedCoreNum = FAST_CORE_0;
        blockFactor = totalNum / usedCoreNum;
        ubFactor = blockFactor;
    } else if (totalNum == FAST_TOTAL_1 && coreNum >= FAST_CORE_1) {
        // 92160 的最优路径：8 核，一次搬运完成。
        usedCoreNum = FAST_CORE_1;
        blockFactor = totalNum / usedCoreNum;
        ubFactor = blockFactor;
    } else {
        // 大 shape：尽量多核，TQue DoubleBuffer 按 4 个 buffer 计算 ubFactor。
        usedCoreNum = CeilDiv(totalNum, GENERIC_BLOCK_TARGET);
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), gmAlignNum);
        usedCoreNum = CeilDiv(totalNum, blockFactor);
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        int64_t maxUbFactor = static_cast<int64_t>(ubSize) / (4 * typeSize);
        maxUbFactor = FloorAlign(maxUbFactor, gmAlignNum);
        if (maxUbFactor > FALLBACK_MAX_UB_FACTOR) {
            maxUbFactor = FALLBACK_MAX_UB_FACTOR;
        }
        OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
        ubFactor = (blockFactor < maxUbFactor) ? blockFactor : maxUbFactor;
    }

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    if (dataType == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling