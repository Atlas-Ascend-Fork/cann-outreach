#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

namespace optiling {
using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TILE_ELEMENTS = 256;

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize < 4 * TILE_ELEMENTS * 4,
        OP_LOGE(context, "insufficient UB"), return ge::GRAPH_FAILED);

    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = inputShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "invalid input size"), return ge::GRAPH_FAILED);
    auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    uint64_t tilingKey = inputDesc->GetDataType() == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_1);

    auto* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), TILE_ELEMENTS);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = TILE_ELEMENTS;
    context->SetBlockDim(CeilDiv(totalNum, blockFactor));
    context->SetTilingKey(tilingKey);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
struct SqrtCompileInfo {};
IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);
} // namespace optiling
