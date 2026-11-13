/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t MAX_CORE_NUM = 8;
constexpr int64_t UB_FACTOR = 128;
constexpr uint32_t TANH_TMP_BYTES = 8 * 1024;

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
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
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
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Only float32 and float16 are supported"),
        return ge::GRAPH_FAILED);

    const gert::Shape shape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = shape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape"),
        return ge::GRAPH_FAILED);

    const uint32_t typeSize = dtype == ge::DT_FLOAT16 ? 2U : 4U;
    const int64_t alignElements = 32 / typeSize;

    // 两个输入 buffer、两个输出 buffer，以及 Tanh 临时空间。
    const uint64_t requiredUb = 4 * UB_FACTOR * typeSize + TANH_TMP_BYTES;
    OP_CHECK_IF(ubSize < requiredUb, OP_LOGE(context, "Insufficient UB space"),
        return ge::GRAPH_FAILED);

    ge::Shape tileShape(std::vector<int64_t>{UB_FACTOR});
    uint32_t maxTmpSize = 0;
    uint32_t minTmpSize = 0;
    AscendC::GetTanhMaxMinTmpSize(
        tileShape, typeSize, false, maxTmpSize, minTmpSize);
    OP_CHECK_IF(TANH_TMP_BYTES < minTmpSize,
        OP_LOGE(context, "Insufficient Tanh temporary space"),
        return ge::GRAPH_FAILED);

    int64_t launchCores = coreNum < MAX_CORE_NUM ? coreNum : MAX_CORE_NUM;
    if (totalNum < MIN_SPLIT_THRESHOLD) {
        launchCores = 1;
    }

    // 每核起点按 32 字节对齐；最后一个核只处理剩余的有效元素。
    const int64_t averageNum = totalNum / launchCores +
        (totalNum % launchCores != 0 ? 1 : 0);
    int64_t blockFactor = (averageNum / alignElements +
        (averageNum % alignElements != 0 ? 1 : 0)) * alignElements;
    if (blockFactor == 0) {
        blockFactor = alignElements;
    }
    const int64_t blockDim = totalNum == 0 ? 1 :
        totalNum / blockFactor + (totalNum % blockFactor != 0 ? 1 : 0);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = UB_FACTOR;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
