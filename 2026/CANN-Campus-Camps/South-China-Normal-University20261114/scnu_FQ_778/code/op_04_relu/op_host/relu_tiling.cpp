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

const uint32_t TILE_NUM = 32;
constexpr int64_t MAX_TOTAL_LENGTH = (1LL << 40);

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    if (x1_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalLength = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        int64_t dim = x1_shape->GetStorageShape().GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (totalLength > MAX_TOTAL_LENGTH / dim) {
            return ge::GRAPH_FAILED;
        }
        totalLength *= dim;
    }
    if (totalLength <= 0 || totalLength > static_cast<int64_t>(0xFFFFFFFFLL)) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->tileNum = TILE_NUM;

    // 配置启动核数
    context->SetBlockDim(8);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
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
