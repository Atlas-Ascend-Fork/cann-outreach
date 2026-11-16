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
constexpr int64_t ALIGN_ELEMENTS = 16; // 32-byte alignment for both fp16/fp32.
constexpr int64_t SMALL_MAX_TILE_BYTES = 32 * 1024;
constexpr int64_t RELU_SMALL_TENSOR_LIMIT = 16384;
// These two parameters are candidates for measurement on the target device.
constexpr int64_t RELU_SMALL_CORE_LIMIT = 8;
constexpr int64_t RELU_LARGE_TILE_BYTES = 32 * 1024;
constexpr uint64_t UB_RESERVED_BYTES = 1024;

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

    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Only float32 and float16 are supported"), return ge::GRAPH_FAILED);
    const auto shape = EnsureNotScalar(inputShape->GetStorageShape());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        OP_CHECK_IF(shape.GetDim(i) < 0, OP_LOGE(context, "Unresolved input shape"),
            return ge::GRAPH_FAILED);
    }
    const int64_t totalNum = shape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input length"), return ge::GRAPH_FAILED);

    // At least about 1024 elements per core, with aligned core boundaries.
    int64_t blockNum = totalNum / MIN_SPLIT_THRESHOLD + (totalNum % MIN_SPLIT_THRESHOLD != 0);
    blockNum = blockNum < 1 ? 1 : (blockNum < coreNum ? blockNum : coreNum);
    if (totalNum <= RELU_SMALL_TENSOR_LIMIT && blockNum > RELU_SMALL_CORE_LIMIT) {
        blockNum = RELU_SMALL_CORE_LIMIT;
    }
    int64_t blockFactor = totalNum / blockNum + (totalNum % blockNum != 0);
    blockFactor = CeilAlign(blockFactor > 0 ? blockFactor : ALIGN_ELEMENTS, ALIGN_ELEMENTS);
    blockNum = totalNum == 0 ? 1 : totalNum / blockFactor + (totalNum % blockFactor != 0);

    // Double-buffered input/output queues: four buffers, no compute scratch.
    const int64_t typeBytes = dtype == ge::DT_FLOAT ? TYPE_SIZE : 2;
    const int64_t bytesPerElement = 4 * typeBytes;
    OP_CHECK_IF(ubSize <= UB_RESERVED_BYTES,
        OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
    int64_t ubFactor = static_cast<int64_t>((ubSize - UB_RESERVED_BYTES) / bytesPerElement);
    ubFactor = FloorAlign(ubFactor, ALIGN_ELEMENTS);
    const int64_t maxTileBytes = totalNum <= RELU_SMALL_TENSOR_LIMIT ?
        SMALL_MAX_TILE_BYTES : RELU_LARGE_TILE_BYTES;
    const int64_t maxTileElements = maxTileBytes / typeBytes;
    ubFactor = ubFactor < maxTileElements ? ubFactor : maxTileElements;
    ubFactor = ubFactor < blockFactor ? ubFactor : blockFactor;
    OP_CHECK_IF(ubFactor < ALIGN_ELEMENTS,
        OP_LOGE(context, "Insufficient UB for one aligned tile"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    // Select the one-shot specialization only when every core fits in one tile.
    const bool useSmall = totalNum > 0 && totalNum <= RELU_SMALL_TENSOR_LIMIT &&
                          blockFactor <= ubFactor;
    if (useSmall) {
        if (dtype == ge::DT_FLOAT16) {
            tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_2);
        } else {
            tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_3);
        }
    } else if (dtype == ge::DT_FLOAT16) {
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
