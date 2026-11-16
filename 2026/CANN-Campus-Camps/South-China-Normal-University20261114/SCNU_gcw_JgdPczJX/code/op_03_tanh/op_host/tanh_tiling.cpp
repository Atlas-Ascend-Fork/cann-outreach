#include "register/op_def_registry.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {
    
const uint32_t TILE_NUM = 8;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (size_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = data_sz;
    tiling->tileNum = TILE_NUM;

    context->SetBlockDim(8);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling