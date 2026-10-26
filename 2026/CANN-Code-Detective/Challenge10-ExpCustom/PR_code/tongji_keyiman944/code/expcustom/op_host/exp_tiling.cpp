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
#ifndef EXP_TILING_PLAN_H
#define EXP_TILING_PLAN_H
#include <cstdint>
struct ExpTilingPlan {
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    uint32_t blockDim = 1;
};
inline bool MakeExpTilingPlan(int64_t total, int64_t typeSize, uint64_t ubSize,
    int64_t coreNum, ExpTilingPlan& plan)
{
    constexpr uint64_t reserveBytes = 1024;
    constexpr int64_t minPerCore = 2048;
    constexpr int64_t maxTile = 16384;
    if (total < 0 || (typeSize != 2 && typeSize != 4) || coreNum <= 0 ||
        coreNum > UINT32_MAX || ubSize <= reserveBytes) {
        return false;
    }
    const int64_t alignment = 32 / typeSize;
    const uint64_t capacity = (ubSize - reserveBytes) / (4 * typeSize) / alignment * alignment;
    if (capacity == 0) {
        return false;
    }
    // Division avoids overflow from total + divisor - 1.
    int64_t blocks = total / minPerCore + (total % minPerCore != 0);
    blocks = blocks < 1 ? 1 : (blocks > coreNum ? coreNum : blocks);
    const int64_t perCore = total / blocks + (total % blocks != 0);
    const int64_t alignedBlocks = perCore / alignment + (perCore % alignment != 0);
    if (alignedBlocks > INT64_MAX / alignment) {
        return false;
    }
    plan.blockFactor = alignedBlocks == 0 ? alignment : alignedBlocks * alignment;
    plan.blockDim = total == 0 ? 1 : static_cast<uint32_t>(
        total / plan.blockFactor + (total % plan.blockFactor != 0));
    const int64_t tileCapacity = capacity > static_cast<uint64_t>(maxTile) ?
        maxTile : static_cast<int64_t>(capacity);
    plan.ubFactor = plan.blockFactor < tileCapacity ? plan.blockFactor : tileCapacity;
    return true;
}
#endif


namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

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

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Exp supports float32 and float16 only"), return ge::GRAPH_FAILED);

    const auto& shape = inputShape->GetStorageShape();
    int64_t totalNum = 1;
    bool empty = false;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "Unresolved input dimension"), return ge::GRAPH_FAILED);
        empty = empty || dim == 0;
    }
    if (empty) {
        totalNum = 0;
    } else {
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            const int64_t dim = shape.GetDim(i);
            OP_CHECK_IF(totalNum > INT64_MAX / dim,
                OP_LOGE(context, "Input element count overflow"), return ge::GRAPH_FAILED);
            totalNum *= dim;
        }
    }
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    ExpTilingPlan plan;
    OP_CHECK_IF(!MakeExpTilingPlan(totalNum, typeSize, ubSize, coreNum, plan),
        OP_LOGE(context, "Invalid Exp tiling resources"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;
    tiling->blockFactor = plan.blockFactor;
    tiling->ubFactor = plan.ubFactor;
    context->SetBlockDim(plan.blockDim);

    const bool fixedShape = totalNum == 16384 && plan.blockFactor == 2048 &&
        plan.ubFactor >= 2048 && plan.blockDim == 8;
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = fixedShape ? GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_2) :
            GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    } else {
        tilingKey = fixedShape ? GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_3) :
            GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
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
