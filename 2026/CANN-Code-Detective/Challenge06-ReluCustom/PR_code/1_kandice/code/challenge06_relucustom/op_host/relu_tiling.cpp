/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t BLOCK_DIM = 8;
constexpr int64_t MAX_TILE_BYTES = 48 * 1024;
constexpr int64_t VECTOR_BYTES = 256;

static inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalLength = inputShape->GetShape().GetShapeSize();
    OP_CHECK_IF(totalLength <= 0, OP_LOGE(context, "totalLength <= 0"), return ge::GRAPH_FAILED);

    const auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();

    int64_t typeSize = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
        context->SetTilingKey(0);
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = 2;
        context->SetTilingKey(1);
    } else {
        OP_LOGE(context, "Relu only supports float32/float16");
        return ge::GRAPH_FAILED;
    }

    const int64_t alignElems = VECTOR_BYTES / typeSize;

    int64_t blockLength = CeilDiv(totalLength, BLOCK_DIM);
    blockLength = CeilDiv(blockLength, alignElems) * alignElems;
    const int64_t usedCoreNum = CeilDiv(totalLength, blockLength);

    int64_t tileLength = std::min(blockLength, MAX_TILE_BYTES / typeSize);
    tileLength = std::max<int64_t>(tileLength, 1);

    const int64_t tileNum = CeilDiv(blockLength, tileLength * BUFFER_NUM);

    ReluTilingData* tilingData = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->totalLength = totalLength;
    tilingData->blockLength = blockLength;
    tilingData->tileNum = tileNum;
    tilingData->tileLength = tileLength;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
