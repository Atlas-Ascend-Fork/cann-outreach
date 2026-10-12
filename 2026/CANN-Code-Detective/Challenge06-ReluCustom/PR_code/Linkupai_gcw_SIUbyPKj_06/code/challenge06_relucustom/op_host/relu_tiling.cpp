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
constexpr int64_t UB_BUFFER_NUM = 4;

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

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape shape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = shape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be greater than 0"), return ge::GRAPH_FAILED);

    const int64_t ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "ubBlockSize must be greater than 0"),
        return ge::GRAPH_FAILED);

    // 每个核至少处理 MIN_SPLIT_THRESHOLD 个元素，且每核数据量按 UB block 对齐。
    int64_t usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), ubBlockSize);
    usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // 一个输入队列和一个输出队列均使用双缓冲，共占用 4 份 UB。
    const int64_t ubCanUse = static_cast<int64_t>(ubSize);
    tiling->ubFactor = FloorAlign(FloorDiv(ubCanUse / TYPE_SIZE, UB_BUFFER_NUM), ubBlockSize);
    if (tiling->ubFactor > tiling->blockFactor) {
        tiling->ubFactor = tiling->blockFactor;
    }
    OP_CHECK_IF(tiling->ubFactor <= 0, OP_LOGE(context, "ubFactor must be greater than 0"),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);

    // 根据输入 dtype 选择 tilingKey
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    uint64_t tilingKey = 0;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "unsupported input dtype");
        return ge::GRAPH_FAILED;
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
