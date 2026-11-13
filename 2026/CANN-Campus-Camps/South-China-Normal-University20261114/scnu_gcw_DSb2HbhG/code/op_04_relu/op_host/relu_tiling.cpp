/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 *
 * Tiling 策略（ReLU 是典型的访存受限（memory bound）算子，切分目标是把 HBM 带宽打满）：
 *   1. 多核切分：totalNum 个元素均分到 coreNum 个 AI Vector 核，得到 blockFactor。
 *      由于 kernel 侧每个核的 Global Memory 起始地址为 blockIdx * blockFactor，
 *      blockFactor 必须按 32B 向上对齐（DataCopy 要求首地址 32B 对齐，且这样可以让
 *      除最后一个核以外的所有核都整块对齐搬运）。
 *   2. 核内切分：UB 中为输入、输出各申请 BUFFER_NUM(=2) 块 buffer（DoubleBuffer），
 *      故单次循环处理的数据量 ubFactor = (UB - 预留) / (2 * BUFFER_NUM * sizeof(T))，
 *      同时受 blockFactor 限制（数据量很小的时候一个核一轮就搬完，不必再切）。
 *   3. 尾块：最后一个核的有效长度可能小于 blockFactor，由 kernel 侧按实际长度搬运。
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
// UB 的最小访问粒度（byte），同时也是 DataCopy 要求的地址对齐粒度
constexpr int64_t UB_BLOCK_SIZE = 32;
// 与 op_kernel/relu.h 中 TQue 的 BUFFER_NUM 保持一致（DoubleBuffer）
constexpr int64_t BUFFER_NUM = 2;
// 输入队列 + 输出队列，各 BUFFER_NUM 块 buffer
constexpr int64_t UB_QUEUE_TIMES = 2 * BUFFER_NUM;
// UB 预留空间：避免把 UB 用满导致 initBuffer 越界
constexpr int64_t UB_RESERVED_SIZE = 8 * 1024;
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
// 输入数据类型对应的字节数
static inline int64_t GetDtypeSize(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
        default:
            return TYPE_SIZE;
    }
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
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    // 输入 shape / dtype 信息（标量（0 维）输入按 1 个元素处理）
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    int64_t dtypeSize = GetDtypeSize(dataType);
    // 32B 对齐所对应的元素个数
    int64_t alignElem = (UB_BLOCK_SIZE / dtypeSize) > 0 ? (UB_BLOCK_SIZE / dtypeSize) : 1;
    // 1. 多核切分：每个核处理的元素个数，按 32B 向上对齐
    int64_t blockFactor = alignElem;
    int64_t usedCoreNum = 1;
    if (totalNum > MIN_SPLIT_THRESHOLD) {
        blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElem);
    } else if (totalNum > 0) {
        // 数据量过小，单核处理即可，省掉多核启动/同步的开销
        blockFactor = CeilAlign(totalNum, alignElem);
    }
    if (blockFactor <= 0) {
        blockFactor = alignElem;
    }
    if (totalNum > 0) {
        usedCoreNum = CeilDiv(totalNum, blockFactor);
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    // 2. 核内切分：受 UB 容量限制，DoubleBuffer 下共占用 2 * BUFFER_NUM 块 buffer
    int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVED_SIZE;
    if (ubBudget <= 0) {
        ubBudget = static_cast<int64_t>(ubSize);
    }
    int64_t ubFactor = FloorAlign(ubBudget / (dtypeSize * UB_QUEUE_TIMES), alignElem);
    if (ubFactor > blockFactor) {
        // 单个核只需要一轮就能处理完，无需再切
        ubFactor = blockFactor;
    }
    if (ubFactor < alignElem) {
        ubFactor = alignElem;
    }
    // 3. 回填 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
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