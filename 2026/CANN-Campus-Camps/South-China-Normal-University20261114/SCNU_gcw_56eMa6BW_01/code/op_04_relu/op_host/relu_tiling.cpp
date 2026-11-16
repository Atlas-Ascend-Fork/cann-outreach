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

#include <algorithm>
#include <cstdint>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
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

    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);

    const gert::Shape inputShape = EnsureNotScalar(inputStorageShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < inputShape.GetDimNum(); ++i) {
        const int64_t dim = inputShape.GetDim(i);
        OP_CHECK_IF(
            dim < 0,
            OP_LOGE(context, "input shape has a negative dimension"),
            return ge::GRAPH_FAILED);
        totalNum *= dim;
    }
    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "totalNum must be positive"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const ge::DataType inputType = inputDesc->GetDataType();
    const int64_t typeSize = (inputType == ge::DT_FLOAT16) ? 2 : 4;

    // 每个使用的矢量核至少处理 1024 个元素。
    const int64_t usedCoreNum =
        std::min<int64_t>(coreNum, (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD);

    tiling->totalNum = totalNum;
    tiling->blockFactor = (totalNum + usedCoreNum - 1) / usedCoreNum;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 双缓冲输入和输出共占用 4 个 UB 缓冲区。
    const int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    const int64_t alignElements = std::max<int64_t>(1, ubBlockSize / typeSize);
    const int64_t maxUbElements =
        static_cast<int64_t>(ubSize) / (BUFFER_NUM * 2 * typeSize);
    const int64_t maxTileElements = std::min(tiling->blockFactor, maxUbElements);

    tiling->ubFactor = (maxTileElements < alignElements)
        ? maxTileElements
        : (maxTileElements / alignElements) * alignElements;

    OP_CHECK_IF(
        tiling->ubFactor <= 0,
        OP_LOGE(context, "ubFactor is not positive"),
        return ge::GRAPH_FAILED);

    const uint64_t tilingKey = (inputType == ge::DT_FLOAT16)
        ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
