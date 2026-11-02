/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

namespace optiling {


static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    if (inputStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();
    uint64_t totalNum = 1;
    for (size_t i = 0; i < inputShape.GetDimNum(); ++i) {
        const int64_t dim = inputShape.GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= static_cast<uint64_t>(dim);
    }

    if (totalNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // Use up to 64 vector cores. The last block can contain fewer elements.
    const uint64_t blockDim = (totalNum < 64) ? totalNum : 64;
    if (blockDim == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t blockFactor = (totalNum + blockDim - 1) / blockDim;
    const uint64_t ubFactor = (blockFactor < 8192) ? blockFactor : 8192;

    auto* rawTiling = context->GetRawTilingData();
    rawTiling->SetDataSize(sizeof(SqrtTilingData));
    auto* tilingData =
        reinterpret_cast<SqrtTilingData*>(rawTiling->GetData());
    tilingData->totalNum = totalNum;
    tilingData->blockFactor = blockFactor;
    tilingData->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(SQRT_TILING_KEY);
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};

IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);

} // namespace optiling
