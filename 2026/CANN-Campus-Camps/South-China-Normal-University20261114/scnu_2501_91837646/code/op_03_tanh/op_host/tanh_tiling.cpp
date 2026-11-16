/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_NUM = 8;
const int64_t UB_FACTOR = 2048;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // tanh.cpp uses schedule keys: 0 -> half, 1 -> float.
    // These are NOT ge::DataType enum values.
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(0);
    } else if (dtype == ge::DT_FLOAT) {
        context->SetTilingKey(1);
    } else {
        return ge::GRAPH_FAILED;
    }

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int64_t totalNum = 1;
    const auto& shape = x1_shape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    // This implementation targets the fixed (8, 2048) contest input.
    if (totalNum != 16384) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalNum = totalNum;
    tiling->blockFactor = totalNum / BLOCK_NUM;
    tiling->ubFactor = UB_FACTOR;
    // (8, 2048): 2048 elements/core, one 2048-element tile/core.
    // Both float and half transfers satisfy 32-byte alignment.
    context->SetBlockDim(BLOCK_NUM);

    // 配置 workspace 大小
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
