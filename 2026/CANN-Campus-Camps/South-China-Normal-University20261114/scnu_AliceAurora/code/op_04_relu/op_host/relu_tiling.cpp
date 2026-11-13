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
constexpr int64_t UB_BLOCK_SIZE = 32;        // UB 最小对齐粒度(Byte)
constexpr int64_t SMALL_TOTAL_NUM = 32768;   // 小 shape 阈值
constexpr int64_t SMALL_BLOCK_NUM = 4;       // 小 shape 核数：block 调度 ~0.26us/个
constexpr int64_t UB_BUDGET_BYTES = 184320;  // 可用 UB 预算 180KB
constexpr int64_t UB_BLOCK_DIV = 4;          // 2 块乒乓缓冲 × 2 层流水深度 → 每块 45KB、约 4 次迭代

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

/* 本地整数工具：向上/向下取整对齐 */
static inline int64_t CeilDivInt(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline int64_t CeilAlignInt(int64_t a, int64_t b) { return CeilDivInt(a, b) * b; }
static inline int64_t FloorAlignInt(int64_t a, int64_t b) { return (a / b) * b; }

static ge::graphStatus GetAivCoreNum(gert::TilingContext* context, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 输入 shape（标量按 {1} 处理），逐维相乘得到总元素数
    const auto* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape& xShape = EnsureNotScalar(xStorageShape->GetStorageShape());
    int64_t totalNum = xShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    // 依据 dtype 确定元素字节数，以及 32B 对齐对应的元素个数
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT) ? TYPE_SIZE : 2;
    int64_t alignNum = UB_BLOCK_SIZE / typeSize;
    OP_CHECK_IF(alignNum <= 0, OP_LOGE(context, "alignNum is invalid"), return ge::GRAPH_FAILED);

    // 启动核数：小 shape 少核（省 block 调度开销）；大 shape 用满 AIV 核
    int64_t blockNum = SMALL_BLOCK_NUM;
    if (totalNum > SMALL_TOTAL_NUM) {
        OP_CHECK_IF(
            GetAivCoreNum(context, blockNum) != ge::GRAPH_SUCCESS,
            OP_LOGE(context, "GetAivCoreNum error"),
            return ge::GRAPH_FAILED);
    }
    if (totalNum < MIN_SPLIT_THRESHOLD) {
        blockNum = 1;
    }

    // 多核切分：按 32B 对齐尽量均分
    int64_t blockFactor = CeilAlignInt(CeilDivInt(totalNum, blockNum), alignNum);
    int64_t usedCoreNum = CeilDivInt(totalNum, blockFactor);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }

    // UB 切分：in-place 乒乓双缓冲（共 2 块），留出流水深度 → 每核约 4 次迭代供填充
    int64_t ubFactor = 0;
    if (totalNum <= SMALL_TOTAL_NUM) {
        ubFactor = blockFactor;
    } else {
        ubFactor = FloorAlignInt(UB_BUDGET_BYTES / (typeSize * UB_BLOCK_DIV), alignNum);
    }
    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
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