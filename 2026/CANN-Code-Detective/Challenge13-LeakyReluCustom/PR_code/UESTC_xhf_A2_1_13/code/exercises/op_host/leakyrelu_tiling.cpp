/*!
 * \file leakyrelu_tiling.cpp
 * \brief Leakyrelu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {
static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
{
    const auto* desc = context->GetInputDesc(0);
    const auto* shape = context->GetInputShape(0);
    auto* info = context->GetPlatformInfo();
    auto* tiling = context->GetTilingData<LeakyreluTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!desc || !shape || !info || !tiling || !workspace) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = desc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(info);
    const int64_t cores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t total = shape->GetStorageShape().GetShapeSize();
    constexpr int64_t align = 256;
    constexpr uint64_t reserve = 1024;
    const int64_t bytes = dtype == ge::DT_FLOAT ? 4 : 2;
    if (cores <= 0 || total <= 0 || ubBytes <= reserve) {
        return ge::GRAPH_FAILED;
    }
    const int64_t block = Ops::Base::CeilAlign(Ops::Base::CeilDiv(total, cores), align);
    const int64_t capacity =
        Ops::Base::FloorAlign(static_cast<int64_t>(ubBytes - reserve) / (4 * bytes), align);
    if (capacity <= 0) {
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    const float* slope = attrs ? attrs->GetAttrPointer<float>(0) : nullptr;
    tiling->totalNum = total;
    tiling->blockFactor = block;
    tiling->ubFactor = block < capacity ? block : capacity;
    tiling->negativeSlope = slope ? *slope : 0.01F;
    context->SetBlockDim(static_cast<uint32_t>(Ops::Base::CeilDiv(total, block)));
    context->SetTilingKey(dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_1));
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLeakyrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LeakyreluCompileInfo {};

IMPL_OP_OPTILING(Leakyrelu).Tiling(LeakyreluTilingFunc).TilingParse<LeakyreluCompileInfo>(TilingParseForLeakyrelu);

} // namespace optiling
