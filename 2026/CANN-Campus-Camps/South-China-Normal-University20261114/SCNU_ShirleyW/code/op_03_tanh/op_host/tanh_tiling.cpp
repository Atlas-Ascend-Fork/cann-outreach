#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

static ge::graphStatus TanhTilingFunc(
    gert::TilingContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto platform =
        platform_ascendc::PlatformAscendC(platformInfo);

    const int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ubSize);

    if (coreNum <= 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    if (inputDesc == nullptr || inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT &&
        dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 2 ||
        shape.GetDim(0) != 8 ||
        shape.GetDim(1) != 2048) {
        return ge::GRAPH_FAILED;
    }

    TanhTilingData* tiling =
        context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    uint32_t blockDim = 8;
    while (blockDim > coreNum) {
        blockDim /= 2;
    }

    tiling->totalNum = shape.GetShapeSize();
    tiling->blockFactor = tiling->totalNum / blockDim;
    tiling->ubFactor = 512;

    // 五块 float 临时缓冲区和一块比较掩码。
    const uint64_t scratchBytes =
        tiling->ubFactor * 5 * sizeof(float) +
        ((tiling->ubFactor + 255) / 256) * 32;

    const uint32_t typeSize =
        dtype == ge::DT_FLOAT ? 4 : 2;

    // 输入、输出队列分别申请两块缓冲区。
    const uint64_t queueBytes =
        tiling->ubFactor * typeSize * 4;

    if (scratchBytes + queueBytes > ubSize) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(blockDim);

    const uint64_t tilingKey =
        dtype == ge::DT_FLOAT16
            ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
            : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);

    context->SetTilingKey(tilingKey);

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;

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
    .TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling