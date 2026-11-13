/*!
 * \file abs_tiling.cpp
 * \brief Abs 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/abs_tiling_data.h"
#include "../op_kernel/abs_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32; // GM/UB 访问对齐字节数

static ge::graphStatus AbsTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE("AbsTiling", "inputDesc is nullptr"), return ge::GRAPH_FAILED);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    AbsTilingData* tiling = context->GetTilingData<AbsTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE("AbsTiling", "tiling data is nullptr"), return ge::GRAPH_FAILED);
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    OP_CHECK_IF(x_shape == nullptr, OP_LOGE("AbsTiling", "input shape is nullptr"), return ge::GRAPH_FAILED);
    int64_t data_sz = 1;
    for (size_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x_shape->GetStorageShape().GetDim(i);
    }
    uint32_t totalLength = static_cast<uint32_t>(data_sz);
    uint32_t elemSize = (DT_X == static_cast<uint32_t>(ge::DT_FLOAT)) ? 4U : 2U;
    uint32_t alignNum = BLOCK_SIZE / elemSize;

    // 专用 shape：8 x 2048 = 16384
    uint32_t totalBlocks = totalLength / alignNum;
    uint32_t coreNum = Ops::Base::GetAivCoreNum(context);
    OP_CHECK_IF(coreNum == 0, OP_LOGE("AbsTiling", "aiv core num is 0"), return ge::GRAPH_FAILED);
    if (coreNum > totalBlocks) {
        coreNum = totalBlocks;
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE("AbsTiling", "core num is 0"), return ge::GRAPH_FAILED);

    tiling->totalLength = totalLength;
    tiling->alignNum = alignNum;
    tiling->blockFactor = (totalBlocks / coreNum) * alignNum; // 小核处理元素数
    tiling->tailBlockNum = totalBlocks % coreNum;             // 大核个数

    context->SetBlockDim(coreNum);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE("AbsTiling", "workspace is nullptr"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAbs([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AbsCompileInfo {};

IMPL_OP_OPTILING(Abs).Tiling(AbsTilingFunc).TilingParse<AbsCompileInfo>(TilingParseForAbs);

} // namespace optiling
