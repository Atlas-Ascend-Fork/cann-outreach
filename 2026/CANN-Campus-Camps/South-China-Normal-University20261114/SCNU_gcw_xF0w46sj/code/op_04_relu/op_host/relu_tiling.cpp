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

    auto inputDescForTiling = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescForTiling);
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape shape = EnsureNotScalar(xShape->GetStorageShape());

    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalNum *= shape.GetDim(i);
    }
    if (totalNum < 0) {
        totalNum = 0;
    }

    // 根据 dtype 计算 32B 对齐元素数（float32=8，float16=16）
    const bool isHalf = (inputDescForTiling->GetDataType() == ge::DT_FLOAT16 ||
                         inputDescForTiling->GetDataType() == ge::DT_BF16);
    const int64_t dtypeSize = isHalf ? 2 : TYPE_SIZE;
    const int64_t alignNum = static_cast<int64_t>(GetUbBlockSize(context)) / dtypeSize;
    const int64_t safeAlignNum = (alignNum > 0) ? alignNum : 1;

    // 多核切分：小 shape 单核，其余按 AIV 核数切分
    int64_t blockDim = coreNum;
    if (totalNum <= MIN_SPLIT_THRESHOLD) {
        blockDim = 1;
    } else if (totalNum < blockDim) {
        blockDim = totalNum;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    int64_t blockFactor = (totalNum > 0) ? CeilDiv(totalNum, blockDim) : 0;

    // UB 空间预算：输入队列 2 份 + 输出队列 2 份
    const int64_t bytesPerElement = 4 * dtypeSize;
    int64_t maxUbElements = FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElement, safeAlignNum);
    if (maxUbElements < safeAlignNum) {
        maxUbElements = safeAlignNum;
    }

    // 单核内按 TILE_NUM 次 UB 循环切分，并保证 32B 对齐、不超过 UB 预算
    constexpr int64_t TILE_NUM = 8;
    int64_t ubFactor = CeilAlign(CeilDiv(blockFactor, TILE_NUM), safeAlignNum);
    if (ubFactor > maxUbElements) {
        ubFactor = maxUbElements;
    }
    if (ubFactor < safeAlignNum) {
        ubFactor = safeAlignNum;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(blockDim);

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
