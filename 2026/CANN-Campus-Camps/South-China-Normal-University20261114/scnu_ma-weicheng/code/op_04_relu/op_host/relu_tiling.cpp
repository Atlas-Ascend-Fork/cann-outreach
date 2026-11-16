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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;  // 单核最小处理元素数：低于该粒度就收缩核数
constexpr int64_t UB_ALIGN_BYTES = 32;  // UB 块大小（字节），搬运/对齐的基本粒度
constexpr int64_t BUFFER_NUM = 2;  // kernel 侧 TQue 的 buffer 份数（与 op_kernel/relu.h 中 BUFFER_NUM 保持一致）
constexpr int64_t QUEUE_NUM = 2;  // 每个 kernel 有 input/output 两个队列，共占用 BUFFER_NUM * 2 份 UB 空间
constexpr int64_t UB_USE_RATIO_NUM = 3;  // UB 使用率上限3/4
constexpr int64_t UB_USE_RATIO_DEN = 4;  // UB 使用率上限

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static inline int64_t GetDtypeSize(ge::DataType dtype)
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

static void SetTilingKey(gert::TilingContext* context)
{
    // 根据输入 dtype 选择 tilingKey：fp16/bf16 走 MODE_0，其余（fp32）走 MODE_1
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // TODO: 实现 Tiling 逻辑
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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    const int64_t dtypeSize = GetDtypeSize(inputDesc->GetDataType());
    const int64_t alignNum = UB_ALIGN_BYTES / dtypeSize;

    // 空张量：不需要启动核，tiling 置 0，kernel 侧循环次数为 0
    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        SetTilingKey(context);
        return ge::GRAPH_SUCCESS;
    }

    // 多核切分：确定 blockFactor 与启动核数量，先按"单核至少 MIN_SPLIT_THRESHOLD 个元素"估算需要的核数，再按实际核数上限收缩
    int64_t usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    // 每核处理的元素数向上对齐到 32B，保证整块搬运都是对齐的快路径
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
    // 对齐之后 blockFactor 可能变大，核数要跟着收缩，避免出现"空核"
    usedCoreNum = CeilDiv(totalNum, blockFactor);

    //核内切分：确定 ubFactor（tile 大小），4 份 buffer：inputQueue(2) + outputQueue(2)，每份 ubFactor 个元素
    int64_t ubAvail = static_cast<int64_t>(ubSize) * UB_USE_RATIO_NUM / UB_USE_RATIO_DEN;
    int64_t ubFactor = FloorAlign(ubAvail / (BUFFER_NUM * QUEUE_NUM * dtypeSize), alignNum);
    if (ubFactor > blockFactor) {
        // 单核数据一 tile 就能装下：不再切分，减少循环/同步开销
        ubFactor = blockFactor;
    }
    if (ubFactor <= 0) {
        // 兜底：UB 极小或数据量极少时，一 tile 搬完
        ubFactor = blockFactor;
    }
    if (blockFactor <= 0) {
        // 理论上不可达，仅做防御
        blockFactor = totalNum;
        ubFactor = totalNum;
        usedCoreNum = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    SetTilingKey(context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
