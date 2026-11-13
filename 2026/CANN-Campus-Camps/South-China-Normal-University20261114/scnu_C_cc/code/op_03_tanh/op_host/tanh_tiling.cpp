
/*!
 * \file tanh_tiling.cpp
 * \brief Tanh operator tiling optimization
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

// Performance parameters
constexpr int64_t PREFERRED_CORES = 16;
constexpr int64_t MAX_TILE_ELEMS = 1024;
constexpr int64_t ALIGN_ELEMS = 16;
constexpr int64_t UB_RESERVE = 8192;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(context, "coreNum is 0"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        currentWorkspace);

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TanhTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // Read input shape
    auto inputShape = context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    // Read input dtype
    auto inputDesc = context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    TanhTilingData* tiling =
        context->GetTilingData<TanhTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    const ge::DataType dtype =
        inputDesc->GetDataType();

    OP_CHECK_IF(
        dtype != ge::DT_FLOAT16 &&
        dtype != ge::DT_FLOAT,
        OP_LOGE(
            context,
            "Only float16 and float32 supported"),
        return ge::GRAPH_FAILED);

    // Total elements
    const int64_t totalNum =
        inputShape->GetStorageShape().GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0 ||
        totalNum % ALIGN_ELEMS != 0,
        OP_LOGE(context, "Invalid input size"),
        return ge::GRAPH_FAILED);

    // ====================================
    // Step 1: Multi-core tiling
    //
    // Prefer 16 AIV cores.
    // ====================================

    int64_t blockDim =
        coreNum < PREFERRED_CORES
            ? coreNum
            : PREFERRED_CORES;

    const int64_t maxUsefulCores =
        totalNum / ALIGN_ELEMS;

    if (blockDim > maxUsefulCores) {
        blockDim = maxUsefulCores;
    }

    const int64_t elementsPerCore =
        (totalNum + blockDim - 1)
        / blockDim;

    const int64_t blockFactor =
        ((elementsPerCore + ALIGN_ELEMS - 1)
         / ALIGN_ELEMS)
        * ALIGN_ELEMS;

    // ====================================
    // Step 2: UB tiling
    //
    // Compatible with current tanh.h:
    //
    // 2 input queue buffers
    // 2 output queue buffers
    // 3 float32 temporary buffers
    // ====================================

    const int64_t inputByteSize =
        (dtype == ge::DT_FLOAT16) ? 2 : 4;

    const int64_t bytesPerElement =
        4 * inputByteSize
        + 3 * sizeof(float);

    OP_CHECK_IF(
        ubSize <= static_cast<uint64_t>(
            UB_RESERVE +
            ALIGN_ELEMS * bytesPerElement),
        OP_LOGE(context, "Insufficient UB"),
        return ge::GRAPH_FAILED);

    int64_t ubFactor =
        (static_cast<int64_t>(ubSize)
         - UB_RESERVE)
        / bytesPerElement;

    ubFactor =
        (ubFactor / ALIGN_ELEMS)
        * ALIGN_ELEMS;

    if (ubFactor > MAX_TILE_ELEMS) {
        ubFactor = MAX_TILE_ELEMS;
    }

    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    // ====================================
    // Step 3: Set tiling parameters
    // ====================================

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(
        static_cast<uint32_t>(blockDim));

    // ====================================
    // Step 4: Select tiling key
    //
    // 0 -> float16
    // 1 -> float32
    // ====================================

    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(
            GET_TPL_TILING_KEY(
                TANH_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(
            GET_TPL_TILING_KEY(
                TANH_TPL_SCH_MODE_1));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh)
    .Tiling(TanhTilingFunc)
    .TilingParse<TanhCompileInfo>(
        TilingParseForTanh);

} // namespace optiling
