/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t ALIGN_BYTES = 32;

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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);

    // 根据输入 dtype 选择 tilingKey：float16/bf16 在 kernel 侧转 float 计算
    const bool isFp16 =
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16);
    const uint64_t schMode = isFp16 ? TANH_TPL_SCH_MODE_0 : TANH_TPL_SCH_MODE_1;
    context->SetTilingKey(GET_TPL_TILING_KEY(schMode));

    // 输入 shape 逐维相乘得到总元素数
    const gert::Shape& xShape = EnsureNotScalar(xShapePtr->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        totalNum *= xShape.GetDim(i);
    }
    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        context->GetRawTilingData()->SetDataSize(sizeof(TanhTilingData));
        return ge::GRAPH_SUCCESS;
    }

    const int64_t typeSize = isFp16 ? 2 : 4;
    const int64_t alignNum = CeilDiv(ALIGN_BYTES, typeSize);

    // 多核切分：每核至少处理 MIN_SPLIT_THRESHOLD 个元素，避免切分过碎
    int64_t blockDim = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    // 每核长度按 32B 对齐，保证 GM 偏移与 DataCopy 长度满足对齐要求
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignNum);
    blockDim = CeilDiv(totalNum, blockFactor);

    // 核内切分：按 UB 容量（预留一倍余量）计算单次处理量，输入/输出双缓冲 + 两个 float 临时张量
    const int64_t perElementBytes = typeSize * 2 * BUFFER_NUM + 2 * TYPE_SIZE;
    int64_t ubElements = static_cast<int64_t>(ubSize / 2) / perElementBytes;
    int64_t ubFactor = FloorAlign(ubElements, alignNum);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
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
    context->GetRawTilingData()->SetDataSize(sizeof(TanhTilingData));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
