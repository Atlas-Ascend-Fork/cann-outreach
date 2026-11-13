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
constexpr int64_t TYPE_SIZE = 4;

// 32B 一个 block，作为 GM <-> UB 搬运的对齐粒度
constexpr int64_t BLOCK_BYTES = 32;
// kernel 侧 TQue 的 buffer 深度（DoubleBuffer）
constexpr int64_t BUFFER_NUM = 2;
// 预留给 TPipe 等系统开销的 UB 空间
constexpr int64_t UB_RESERVE_BYTES = 1024;

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

static inline int64_t ReluGetTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
            return 4;
        default:
            return TYPE_SIZE;
    }
}

// 每核目标元素数：数据越小，单核固定开销占比越高，应少开核；数据越大越靠多核并行。
static inline int64_t ReluTargetElemsPerCore(int64_t totalNum)
{
    constexpr int64_t SMALL_TOTAL = 32768;      // <= 32K 元素
    constexpr int64_t MEDIUM_TOTAL = 262144;    // <= 256K 元素
    if (totalNum <= SMALL_TOTAL) {
        return 4096;
    }
    if (totalNum <= MEDIUM_TOTAL) {
        return 8192;
    }
    return 32768;
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

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const int64_t typeSize = ReluGetTypeSize(dtype);

    // 32B 对齐对应的元素个数：fp32 为 8，fp16 为 16
    const int64_t alignNum = (typeSize > BLOCK_BYTES) ? 1 : (BLOCK_BYTES / typeSize);

    // 按“每核目标元素数”决定核数：小数据少开核，避免每核固定开销拖慢整体
    const int64_t targetElems = ReluTargetElemsPerCore(totalNum);
    int64_t usedCoreNum = CeilDiv(totalNum, targetElems);
    usedCoreNum = (usedCoreNum < 1) ? 1 : usedCoreNum;
    usedCoreNum = (usedCoreNum > coreNum) ? coreNum : usedCoreNum;

    // 均分到各核，并保证每个核的 GM 起始地址满足 32B 对齐
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
    usedCoreNum = (blockFactor == 0) ? 1 : CeilDiv(totalNum, blockFactor);
    usedCoreNum = (usedCoreNum < 1) ? 1 : usedCoreNum;
    usedCoreNum = (usedCoreNum > coreNum) ? coreNum : usedCoreNum;

    // 单次搬运的元素个数：UB 内 in/out 各 BUFFER_NUM 块，按 32B 向下对齐
    int64_t availUb = static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;
    if (availUb <= 0) {
        availUb = static_cast<int64_t>(ubSize);
    }
    int64_t ubFactor = FloorAlign(availUb / (typeSize * BUFFER_NUM * 2), alignNum);
    if (blockFactor > 0 && ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 启动核数量
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
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