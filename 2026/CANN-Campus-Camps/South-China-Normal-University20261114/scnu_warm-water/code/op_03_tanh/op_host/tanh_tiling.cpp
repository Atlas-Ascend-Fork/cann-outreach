/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_COUNT = 8;
constexpr uint32_t TILE_NUM = 1;
constexpr uint32_t BUFFER_NUM = 1;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // Match the platform kernel entry: mode 0 is half, mode 1 is float.
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t schMode = dtype == ge::DT_FLOAT16 ? TANH_TPL_SCH_MODE_0 : TANH_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const auto& shape = x1_shape->GetStorageShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalLength *= static_cast<uint32_t>(shape.GetDim(i));
    }
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;
    context->SetBlockDim(BLOCK_COUNT);

    const uint32_t tileLength = totalLength / BLOCK_COUNT / TILE_NUM / BUFFER_NUM;
    const uint32_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const ge::Shape tileShape(std::vector<int64_t>{tileLength});
    uint32_t maxTmpBytes = 0;
    uint32_t minTmpBytes = 0;
    AscendC::GetTanhMaxMinTmpSize(tileShape, typeSize, false, maxTmpBytes, minTmpBytes);
    tiling->tmpBytes = (maxTmpBytes + 31U) / 32U * 32U;

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
