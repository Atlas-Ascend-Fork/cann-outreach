#include "register/op_def_registry.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    constexpr uint32_t kBlockDim = 8;
    constexpr int64_t kMaxTileLength = 4096;

    const auto dtype = context->GetInputDesc(0)->GetDataType();
    const uint32_t schMode = (dtype == ge::DT_FLOAT16)
        ? RELU_TPL_SCH_MODE_0
        : RELU_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    const auto& shape = context->GetInputShape(0)->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }

    const int64_t blockFactor = totalNum / kBlockDim;
    const int64_t halfBlock = blockFactor / 2;

    auto* tiling = context->GetTilingData<ReluTilingData>();
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = (halfBlock < kMaxTileLength)
        ? halfBlock
        : kMaxTileLength;

    context->SetBlockDim(kBlockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

}  // namespace optiling