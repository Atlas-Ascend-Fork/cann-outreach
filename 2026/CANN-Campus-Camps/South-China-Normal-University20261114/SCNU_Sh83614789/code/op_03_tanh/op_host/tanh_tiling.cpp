#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

const uint32_t TILE_NUM = 8;
const uint32_t BLOCK_DIM = 8;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int64_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = static_cast<uint32_t>(data_sz);
    tiling->tileNum = TILE_NUM;

    context->SetBlockDim(BLOCK_DIM);

    uint32_t dtype = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
    uint32_t schMode = (dtype == static_cast<uint32_t>(ge::DT_FLOAT16)) ? TANH_TPL_SCH_MODE_1 : TANH_TPL_SCH_MODE_0;
    uint64_t tilingKey = GET_TPL_TILING_KEY(schMode);
    context->SetTilingKey(tilingKey);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
