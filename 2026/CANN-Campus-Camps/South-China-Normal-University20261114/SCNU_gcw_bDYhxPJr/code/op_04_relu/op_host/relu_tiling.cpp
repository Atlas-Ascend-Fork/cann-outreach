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

// kernel 侧每个队列的 buffer 块数（双缓冲）
constexpr int64_t BUFFER_NUM = 2;
// kernel 侧 UB 总占用 = ubFactor * typeSize * (输入队列 2 块 + 输出队列 2 块)
constexpr int64_t UB_QUEUE_FACTOR = BUFFER_NUM * 2;
// UB 只使用 3/4，留出余量给框架开销
constexpr int64_t UB_RATIO_NUM = 3;
constexpr int64_t UB_RATIO_DEN = 4;
// UB 的 block 粒度固定 32 字节
constexpr int64_t UB_BLOCK_BYTES = 32;

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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---------- 1. dtype -> typeSize / tilingKey ----------
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    const bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    const int64_t typeSize = isFp16 ? (TYPE_SIZE / 2) : TYPE_SIZE;

    // 用 if/else 分别赋值，避免宏嵌在三元表达式里
    uint64_t tilingKey;
    if (isFp16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    // ---------- 2. 展平 shape，得到元素总数 ----------
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape xShape = EnsureNotScalar(inputShapePtr->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        totalNum *= xShape.GetDim(i);
    }

    // 空 tensor 直接返回，避免除零
    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 1;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // ---------- 3. 多核切分 ----------
    const int64_t alignElem = UB_BLOCK_BYTES / typeSize;    // 32B 折算成元素个数
    int64_t usedCoreNum = coreNum;
    if (totalNum < MIN_SPLIT_THRESHOLD * coreNum) {         // 小 shape 不铺满核
        usedCoreNum = (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }
    }
    // blockFactor 向上对齐到 alignElem，保证每个核的数据量都是 32B 整数倍
    int64_t blockFactor = (totalNum + usedCoreNum - 1) / usedCoreNum;
    blockFactor = (blockFactor + alignElem - 1) / alignElem * alignElem;
    usedCoreNum = (totalNum + blockFactor - 1) / blockFactor;

    // ---------- 4. UB 切分 ----------
    const int64_t ubUsable = static_cast<int64_t>(ubSize) * UB_RATIO_NUM / UB_RATIO_DEN;
    int64_t ubFactor = (ubUsable / typeSize) / UB_QUEUE_FACTOR;
    ubFactor = ubFactor / alignElem * alignElem;
    if (ubFactor < alignElem) {
        ubFactor = alignElem;
    }
    if (ubFactor > blockFactor) {                           // 单核一次搬完
        ubFactor = blockFactor;
    }

    // ---------- 5. 落 tiling 数据 + 启动核数 ----------
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
