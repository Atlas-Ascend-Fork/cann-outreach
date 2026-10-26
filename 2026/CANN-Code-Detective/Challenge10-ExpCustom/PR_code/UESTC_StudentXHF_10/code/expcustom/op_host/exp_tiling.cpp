/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {


static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    if (inputShape == nullptr || inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const int64_t count = inputShape->GetStorageShape().GetShapeSize();
    // The challenge specifies (8, 2048). Every tile is 32-byte aligned
    // for both supported dtypes, and uses at most 2 KiB of UB.
    constexpr uint64_t TILE_ELEMENTS = 256;
    if (count <= 0 || count % TILE_ELEMENTS != 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockDim = 8;
    while (blockDim > 1 && count % (blockDim * TILE_ELEMENTS) != 0) {
        blockDim /= 2;
    }
    auto* tiling = context->GetTilingData<ExpTilingData>();
    tiling->totalNum = static_cast<uint64_t>(count);
    tiling->blockFactor = tiling->totalNum / blockDim;
    tiling->ubFactor = TILE_ELEMENTS;
    const uint64_t tilingKey = dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
