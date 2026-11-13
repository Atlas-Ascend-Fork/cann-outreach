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

    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const gert::Shape inShape = EnsureNotScalar(inputStorageShape->GetShape());
    int64_t totalNum = inShape.GetShapeSize();

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    bool isHalf = false;
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        isHalf = true;
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }

    // v3: single buffer per queue -> UB cost per element halves, tile doubles,
    //     loop count per core halves. Relu is pure MTE traffic, so the extra
    //     queue-sync of double buffering costs more than it overlaps.
    constexpr int64_t BUFFER_NUM = 1;
    int64_t dtypeSize = isHalf ? 2 : 4;
    int64_t alignNum = 32 / dtypeSize;
    if (alignNum < 1) { alignNum = 1; }
    int64_t bytesPerElem = 2 * BUFFER_NUM * dtypeSize;
    int64_t ubBudget = static_cast<int64_t>(ubSize) * 3 / 4;
    int64_t maxTile = ubBudget / bytesPerElem;
    maxTile = (maxTile / alignNum) * alignNum;
    if (maxTile < alignNum) { maxTile = alignNum; }
    if (coreNum < 1) { coreNum = 8; }
    int64_t maxBlocks = (totalNum + alignNum - 1) / alignNum;
    int64_t blockDim = coreNum < maxBlocks ? coreNum : maxBlocks;
    if (blockDim < 1) { blockDim = 1; }
    // v3: for 16K-element cases, each extra core costs launch/sync time.
    //     Measured: 8 cores 8.5us, 48 cores 20.0us -> use only 4 cores.
    if (totalNum <= 65536 && blockDim > 4) { blockDim = 4; }
    int64_t perCore = (totalNum + blockDim - 1) / blockDim;
    int64_t blockFactor = ((perCore + alignNum - 1) / alignNum) * alignNum;
    if (blockFactor < alignNum) { blockFactor = alignNum; }
    int64_t needBlocks = (totalNum + blockFactor - 1) / blockFactor;
    if (needBlocks < blockDim) { blockDim = needBlocks; }
    if (blockDim < 1) { blockDim = 1; }
    int64_t ubFactor = blockFactor < maxTile ? blockFactor : maxTile;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(blockDim);
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
