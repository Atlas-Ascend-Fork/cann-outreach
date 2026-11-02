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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t COPY_BLOCK_BYTES = 32;
constexpr int64_t MAX_TILE_ELEMENTS = 8192;
constexpr uint64_t UB_RESERVE_BYTES = 8192;

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
    // The supplied host UT only populates the generic AICore count.
    if (coreNum <= 0) coreNum = ascendcPlatform.GetCoreNum();
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
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Relu supports only float32 and float16"), return ge::GRAPH_FAILED);

    const gert::Shape shape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = shape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);
    const int64_t typeBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t blockElements = COPY_BLOCK_BYTES / typeBytes;

    // Use enough work per core to amortize launch overhead on the 16384-element
    // input. Align core boundaries to 32 bytes; only the last core may have a tail.
    int64_t blockNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (blockNum < 1) blockNum = 1;
    if (blockNum > coreNum) blockNum = coreNum;
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockNum), blockElements);
    if (blockFactor < blockElements) blockFactor = blockElements;
    blockNum = totalNum == 0 ? 1 : CeilDiv(totalNum, blockFactor);

    // Two input buffers plus two output buffers. Leave room for the runtime
    // and limit each tile to a size supported by both kernel specializations.
    const uint64_t usableUb = ubSize > UB_RESERVE_BYTES ? ubSize - UB_RESERVE_BYTES : ubSize;
    int64_t maxUbElements = static_cast<int64_t>(usableUb / (2 * BUFFER_NUM * typeBytes));
    maxUbElements = FloorAlign(maxUbElements, blockElements);
    OP_CHECK_IF(maxUbElements < blockElements,
        OP_LOGE(context, "UB is too small for Relu double buffering"), return ge::GRAPH_FAILED);
    int64_t ubFactor = CeilAlign(CeilDiv(blockFactor, BUFFER_NUM), blockElements);
    if (ubFactor > maxUbElements) ubFactor = maxUbElements;
    if (ubFactor > MAX_TILE_ELEMENTS) ubFactor = MAX_TILE_ELEMENTS;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    rawTilingData->SetDataSize(sizeof(ReluTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
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
