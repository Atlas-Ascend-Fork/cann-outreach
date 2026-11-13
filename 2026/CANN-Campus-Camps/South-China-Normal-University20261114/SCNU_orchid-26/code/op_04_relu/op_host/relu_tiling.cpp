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
    // TODO: 实现 Tiling 逻辑
    // 平台信息在下方获取后统一计算 tiling 参数。
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
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();

    int64_t totalNum = 1;
    for (size_t i = 0; i < inputShape.GetDimNum(); ++i) {
        totalNum *= inputShape.GetDim(i);
    }

    auto inputDescForTiling = context->GetInputDesc(0);
    int64_t typeSize = TYPE_SIZE;
    if (inputDescForTiling != nullptr &&
        (inputDescForTiling->GetDataType() == ge::DT_FLOAT16 ||
         inputDescForTiling->GetDataType() == ge::DT_BF16)) {
        typeSize = 2;
    }

    uint32_t ubBlockSize = GetUbBlockSize(context);
    int64_t alignNum = static_cast<int64_t>(ubBlockSize) / typeSize;
    if (alignNum <= 0) {
        alignNum = 1;
    }

    // BUFFER_NUM = 2，且输入/输出各一个队列，按 2 * 2 * ubFactor * sizeof(T) 估算。
    int64_t ubFactor = static_cast<int64_t>(ubSize) / (4 * typeSize);
    ubFactor = FloorAlign(ubFactor, alignNum);
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    if (blockFactor < 1) {
        blockFactor = 1;
    }
    blockFactor = CeilAlign(blockFactor, alignNum);

    int64_t blockDim = CeilDiv(totalNum, blockFactor);
    if (blockDim < 1) {
        blockDim = 1;
    }
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }

    struct ReluTilingSetter {
        gert::TilingContext* ctx;
        ReluTilingData* data;
        int64_t total;
        int64_t block;
        int64_t ub;
        int64_t dim;
        ~ReluTilingSetter()
        {
            data->totalNum = total;
            data->blockFactor = block;
            data->ubFactor = ub;
            ctx->SetBlockDim(static_cast<uint32_t>(dim));
        }
    } setter{context, tiling, totalNum, blockFactor, ubFactor, blockDim};

    tiling->totalNum = 0;
    tiling->blockFactor = 1;
    tiling->ubFactor = 0;

    context->SetBlockDim(1);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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