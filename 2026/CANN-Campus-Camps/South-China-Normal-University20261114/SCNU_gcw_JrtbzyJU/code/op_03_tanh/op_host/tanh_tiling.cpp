#include "register/op_def_registry.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    constexpr uint32_t kBlockDim = 8;
    constexpr int64_t kUbFactor = 128;

    const auto dtype = context->GetInputDesc(0)->GetDataType();
    const uint32_t schMode = (dtype == ge::DT_FLOAT16)
        ? TANH_TPL_SCH_MODE_0
        : TANH_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    const auto& shape = context->GetInputShape(0)->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }

    auto* tiling = context->GetTilingData<TanhTilingData>();
    tiling->totalNum = totalNum;
    tiling->blockFactor = totalNum / kBlockDim;
    tiling->ubFactor = kUbFactor;

    context->SetBlockDim(kBlockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
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

}  // namespace optiling