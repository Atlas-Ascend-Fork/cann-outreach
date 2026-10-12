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

// DataCopy 要求 32B 对齐；in/out 两个队列各 BUFFER_NUM 份 UB buffer（与 kernel 侧保持一致）
constexpr int64_t DATA_COPY_ALIGN_SIZE = 32;
constexpr int64_t QUEUE_NUM = 2;
constexpr int64_t BUFFER_NUM = 2;
// UB 预留量：TPipe 自身有少量开销，不做满配，避免大 shape 下 InitBuffer 超 UB
constexpr int64_t UB_RESERVED_SIZE = 8 * 1024;

static const gert::Shape g_vec_1_shape = {1};

// TilingContext::GetInputShape 在不同 CANN 版本分别返回 gert::Shape* / gert::StorageShape*，
// 用重载屏蔽版本差异
static inline const gert::Shape& ToShape(const gert::StorageShape* shapePtr)
{
    return shapePtr->GetStorageShape();
}

static inline const gert::Shape& ToShape(const gert::Shape* shapePtr)
{
    return *shapePtr;
}

static inline int64_t GetTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
            return TYPE_SIZE;
        default:
            return TYPE_SIZE;
    }
}

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
    uint64_t ubSize = 0U;
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

    // 1. 输入 shape -> 元素总数（标量视为 1 维 1 元素）
    const auto* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape xShape = EnsureNotScalar(ToShape(xShapePtr));
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        totalNum *= xShape.GetDim(i);
    }
    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "invalid totalNum %ld", totalNum),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const int64_t typeSize = GetTypeSize(dtype);
    const int64_t alignNum = (typeSize > 0) ? (DATA_COPY_ALIGN_SIZE / typeSize) : 1;

    // 2. 核数切分：小 shape 单核，其余按 32B 对齐粒度铺核，最多占满 AI Core
    int64_t blockDim = coreNum;
    if (totalNum < MIN_SPLIT_THRESHOLD) {
        blockDim = 1;
    } else {
        const int64_t maxBlock = CeilDiv(totalNum, alignNum);
        blockDim = (coreNum < maxBlock) ? coreNum : maxBlock;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    // 3. 每核处理长度（向上 32B 对齐，保证 DataCopy 地址对齐）
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignNum);
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }

    // 4. UB 内切分：向下 32B 对齐，保证 2 队列 * 双缓冲不超 UB
    const int64_t ubUsable = (static_cast<int64_t>(ubSize) > UB_RESERVED_SIZE) ?
        (static_cast<int64_t>(ubSize) - UB_RESERVED_SIZE) : static_cast<int64_t>(ubSize);
    const int64_t ubMaxNum = ubUsable / (QUEUE_NUM * BUFFER_NUM * typeSize);
    int64_t ubFactor = FloorAlign(ubMaxNum, alignNum);
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 5. 根据输入 dtype 选择 tilingKey：half/bf16 -> mode0，float -> mode1
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
