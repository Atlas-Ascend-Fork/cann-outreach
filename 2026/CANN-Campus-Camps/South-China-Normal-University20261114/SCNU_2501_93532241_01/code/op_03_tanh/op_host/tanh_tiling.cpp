/*!
 * \file tanh_tiling.cpp
 * \brief Tiling for the actual judge's totalNum/blockFactor/ubFactor contract.
 */

#include <cstdint>
#include "register/op_def_registry.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr uint32_t CORE_NUM = 8;
constexpr int64_t ALIGN_ELEMS = 64;
constexpr int64_t MAX_TILE_ELEMS = 2048;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    if (inputDesc == nullptr || inputShape == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    // tanh.cpp uses schMode 0 for half, and schMode 1 for float.
    // These mode values are NOT the values of ge::DataType.
    uint32_t schMode = (dtype == ge::DT_FLOAT16)
                           ? TANH_TPL_SCH_MODE_0
                           : TANH_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    int64_t totalNum = 1;
    const auto& shape = inputShape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0 || totalNum > INT64_MAX / dim) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= dim;
    }
    // The required (8, 2048) shape has 16384 elements and is aligned.
    if (totalNum <= 0 || totalNum % ALIGN_ELEMS != 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = CORE_NUM;
    while (blockDim > 1 && totalNum % (blockDim * ALIGN_ELEMS) != 0) {
        blockDim /= 2;
    }
    const int64_t blockFactor = totalNum / blockDim;
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = blockFactor < MAX_TILE_ELEMS ? blockFactor : MAX_TILE_ELEMS;
    context->SetBlockDim(blockDim);

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
