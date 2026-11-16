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
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

const uint32_t TILE_NUM = 8;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（schMode=1 -> float，schMode=0 -> half）
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    uint32_t schMode = (dtype == ge::DT_FLOAT) ? TANH_TPL_SCH_MODE_1 : TANH_TPL_SCH_MODE_0;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    // 1) totalNum
    const auto& xShape = x1_shape->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        totalNum *= xShape.GetDim(i);
    }

    // 2) 选 blockDim：要求 totalNum 能被整除，且每核长度能被 TILE_NUM × BUFFER_NUM × 32B对齐元素数 整除
    int32_t dtypeSize = ge::GetSizeByDataType(dtype);
    if (dtypeSize <= 0) { dtypeSize = 4; }
    int64_t alignElems = 32 / dtypeSize;
    int64_t perCoreUnit = static_cast<int64_t>(TILE_NUM) * 2 * alignElems;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    if (coreNum <= 0) { coreNum = 1; }
    if (coreNum > totalNum) { coreNum = totalNum; }

    int64_t blockDim = 0;
    for (int64_t dim = coreNum; dim >= 1; --dim) {
        if (totalNum % dim != 0) { continue; }
        if ((totalNum / dim) % perCoreUnit != 0) { continue; }
        blockDim = dim;
        break;
    }
    if (blockDim <= 0) { blockDim = 1; }

    int64_t blockFactor = totalNum / blockDim;
    int64_t ubFactor = blockFactor / (static_cast<int64_t>(TILE_NUM) * 2);

    // 3) 填充 TilingData
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 4) 配置启动核数
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

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