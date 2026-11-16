#include "register/op_def_registry.h"

#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

static ge::graphStatus TanhTilingFunc(
    gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto dtype = inputDesc->GetDataType();

    // 必须对应现有 tanh.cpp 的分支：
    // schMode = 0 -> Tanh<half>
    // schMode = 1 -> Tanh<float>
    uint32_t schMode;

    if (dtype == ge::DT_FLOAT16) {
        schMode = TANH_TPL_SCH_MODE_0;
    } else if (dtype == ge::DT_FLOAT) {
        schMode = TANH_TPL_SCH_MODE_1;
    } else {
        return ge::GRAPH_FAILED;
    }

    ASCENDC_TPL_SEL_PARAM(context, schMode);

    TanhTilingData* tiling =
        context->GetTilingData<TanhTilingData>();

    tiling->totalLength = static_cast<uint32_t>(
        context->GetInputShape(0)
            ->GetStorageShape()
            .GetShapeSize());

    // 配套上一版 tanh.h：每核一次处理全部数据。
    tiling->tileNum = 1;

    // 固定输入 (8, 2048)，8 核各处理 2048 个元素。
    context->SetBlockDim(8);

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh)
    .Tiling(TanhTilingFunc)
    .TilingParse<TanhCompileInfo>(TilingParseForTanh);

}  // namespace optiling