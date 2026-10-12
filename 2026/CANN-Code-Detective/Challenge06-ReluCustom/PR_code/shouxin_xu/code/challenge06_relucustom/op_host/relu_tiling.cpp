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

const uint32_t TILE_NUM = 16;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 根据输入数据类型选择 float/float16 对应的 kernel 调度模式。
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }

    // 将输入 shape 各维相乘，得到需要处理的总元素数。
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int32_t i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        totalLength *= xShape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    // 题目输入共 16384 个元素，按 8 个 Vector Core 均分处理。
    context->SetBlockDim(8);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
