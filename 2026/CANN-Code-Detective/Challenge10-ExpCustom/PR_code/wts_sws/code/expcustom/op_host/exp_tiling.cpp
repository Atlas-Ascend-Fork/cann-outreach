/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
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

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

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

    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    if (inputDesc == nullptr || inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dataType = inputDesc->GetDataType();
    const int64_t typeSize = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) ? 2 : TYPE_SIZE;
    const int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    if (ubBlockSize <= 0 || typeSize <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t alignElements = std::max<int64_t>(1, ubBlockSize / typeSize);

    const gert::Shape storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        const int64_t dim = storageShape.GetDim(i);
        if (dim < 0) {
            return ge::GRAPH_FAILED;
        }
        if (dim == 0) {
            totalNum = 0;
            break;
        }
        if (totalNum > std::numeric_limits<int64_t>::max() / dim) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= dim;
    }

    int64_t blockFactor = 1;
    int64_t blockDim = 1;
    if (totalNum > 0) {
        const int64_t targetCoreNum = std::min<int64_t>(coreNum, totalNum);
        const int64_t elementsPerCore = CeilDiv(totalNum, targetCoreNum);
        blockFactor = CeilDiv(elementsPerCore, alignElements) * alignElements;
        if (blockFactor > totalNum) {
            blockFactor = totalNum;
        }
        blockDim = CeilDiv(totalNum, blockFactor);
    }

    // The two queues each own BUFFER_NUM buffers; reserve half of UB for runtime use.
    const int64_t maxUbElements = static_cast<int64_t>((ubSize / 2) / (4 * typeSize));
    int64_t ubFactor = 1;
    if (totalNum > 0) {
        if (maxUbElements <= 0) {
            return ge::GRAPH_FAILED;
        }
        ubFactor = std::min(blockFactor, maxUbElements);
        if (ubFactor >= alignElements) {
            ubFactor = (ubFactor / alignElements) * alignElements;
        }
        if (ubFactor <= 0) {
            return ge::GRAPH_FAILED;
        }
    }

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    uint64_t tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
