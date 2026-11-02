/*!
 * \file relu_tiling.cpp
 * \brief ReLU tiling for FP16 / FP32 ND tensors.
 */

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t MAX_CORES = 8;  // Benchmark 4 / 8 / 16 / 32 on the target NPU.
constexpr int64_t ALIGN_BYTES = 512;
constexpr uint64_t UB_RESERVE_BYTES = 4096;
constexpr uint64_t MAX_TILE_BYTES = 32 * 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    return in_shape.GetDimNum() == 0 ? g_vec_1_shape : in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                      uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"),
                return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const ge::DataType dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
                OP_LOGE(context, "Only float32 and float16 are supported"),
                return ge::GRAPH_FAILED);

    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid shape size"),
                return ge::GRAPH_FAILED);
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignElems = ALIGN_BYTES / typeSize;

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = 1;
    tiling->ubFactor = 0;
    uint32_t blockDim = 1;

    if (totalNum != 0) {
        OP_CHECK_IF(ubSize <= UB_RESERVE_BYTES,
                    OP_LOGE(context, "Insufficient UB size"), return ge::GRAPH_FAILED);

        // Keep small slices on one core; bound the core count for this small workload.
        const int64_t workCores = std::max<int64_t>(1, totalNum / MIN_SPLIT_THRESHOLD);
        const int64_t targetCores = std::min(std::min(coreNum, MAX_CORES), workCores);
        const int64_t totalUnits = totalNum / alignElems + (totalNum % alignElems != 0);
        const int64_t unitsPerCore = totalUnits / targetCores + (totalUnits % targetCores != 0);
        const int64_t blockFactor = unitsPerCore * alignElems;
        blockDim = static_cast<uint32_t>(
            totalNum / blockFactor + (totalNum % blockFactor != 0));

        // Two queues with two physical buffers each: 4 * ubFactor * typeSize.
        const uint64_t inputTileBytes = std::min<uint64_t>(
            (ubSize - UB_RESERVE_BYTES) / 4, MAX_TILE_BYTES);
        const int64_t ubLimit = static_cast<int64_t>(inputTileBytes / ALIGN_BYTES) * alignElems;
        OP_CHECK_IF(ubLimit == 0, OP_LOGE(context, "UB tile capacity is 0"),
                    return ge::GRAPH_FAILED);

        tiling->blockFactor = blockFactor;
        tiling->ubFactor = std::min(blockFactor, ubLimit);
    }

    context->SetBlockDim(blockDim);
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
