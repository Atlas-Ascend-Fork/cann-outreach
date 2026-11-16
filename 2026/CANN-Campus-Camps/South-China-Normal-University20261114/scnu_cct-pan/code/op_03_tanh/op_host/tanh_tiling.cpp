/*!
 * \file tanh_tiling.cpp
 * \brief Fixed-shape Tanh tiling for float32/float16.
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {
static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    auto shape = context->GetInputShape(0);
    auto desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const auto& dims = shape->GetStorageShape();
    OP_CHECK_IF(dims.GetDimNum() != 2 || dims.GetDim(0) != 8 || dims.GetDim(1) != 2048,
                OP_LOGE(context, "Expected shape (8, 2048)"), return ge::GRAPH_FAILED);
    const auto dtype = desc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
                OP_LOGE(context, "Only float32 and float16 supported"), return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    int64_t availableCores = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(availableCores <= 0 || ubSize <= 4096,
                OP_LOGE(context, "Invalid platform resources"), return ge::GRAPH_FAILED);
    int64_t cores = 8;
    while (cores > availableCores) { cores /= 2; }
    const int64_t totalNum = 16384;
    const int64_t blockFactor = totalNum / cores;
    const uint64_t typeBytes = dtype == ge::DT_FLOAT ? 4 : 2;
    // Two input/output queue buffers, plus two float temporary tensors.
    const uint64_t bytesPerElement = 4 * typeBytes + 2 * sizeof(float);
    int64_t ubFactor = blockFactor;
    while (ubFactor >= 128 && static_cast<uint64_t>(ubFactor) * bytesPerElement + 4096 > ubSize) {
        ubFactor /= 2;
    }
    OP_CHECK_IF(ubFactor < 128, OP_LOGE(context, "Insufficient UB"), return ge::GRAPH_FAILED);
    auto tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(cores);
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ?
        GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1));
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{ return ge::GRAPH_SUCCESS; }
struct TanhCompileInfo {};
IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);
} // namespace optiling