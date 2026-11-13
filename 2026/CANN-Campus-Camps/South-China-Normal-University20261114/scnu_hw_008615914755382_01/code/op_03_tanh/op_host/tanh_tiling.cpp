/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include <cstdint>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr uint32_t TILE_NUM = 8;
constexpr uint32_t BLOCK_DIM = 8;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    const gert::StorageShape* inputShape = context->GetInputShape(0);

    if (inputDesc == nullptr || inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    // 与tanh.cpp中的实例选择保持一致：
    // schMode = 0：half
    // schMode = 1：float
    uint32_t schMode = dtype == ge::DT_FLOAT16
        ? TANH_TPL_SCH_MODE_0
        : TANH_TPL_SCH_MODE_1;

    ASCENDC_TPL_SEL_PARAM(context, schMode);

    TanhTilingData* tiling =
        context->GetTilingData<TanhTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // ND格式，逐维计算总元素数
    const auto& shape = inputShape->GetStorageShape();
    uint64_t totalLength = 1;

    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);

        if (dim <= 0 ||
            totalLength > 0xFFFFFFFFULL / static_cast<uint64_t>(dim)) {
            return ge::GRAPH_FAILED;
        }

        totalLength *= static_cast<uint64_t>(dim);
    }

    // 当前实现采用均匀分块
    constexpr uint32_t LOOP_COUNT = TILE_NUM * 2;

    if (totalLength % (BLOCK_DIM * LOOP_COUNT) != 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t tileLength =
        static_cast<uint32_t>(totalLength / BLOCK_DIM / LOOP_COUNT);

    const uint32_t typeSize = dtype == ge::DT_FLOAT ? 4U : 2U;

    // 连续DataCopy要求搬运长度满足32字节对齐
    if (tileLength == 0 || (tileLength * typeSize) % 32 != 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(BLOCK_DIM);

    // 对应tanh_tiling_data.h中的实际字段
    tiling->totalNum = static_cast<int64_t>(totalLength);
    tiling->blockFactor =
        static_cast<int64_t>(totalLength / BLOCK_DIM);
    tiling->ubFactor = static_cast<int64_t>(tileLength);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh)
    .Tiling(TanhTilingFunc)
    .TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
