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


static ge::graphStatus AbsTilingFunc(gert::TilingContext* context)
{
    const auto* desc = context->GetInputDesc(0);
    const auto* shape = context->GetInputShape(0);
    auto* platformInfo = context->GetPlatformInfo();
    auto* tiling = context->GetTilingData<AbsTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!desc || !shape || !platformInfo || !tiling || !workspace) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = desc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
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
    const int64_t capacity = Ops::Base::FloorAlign(static_cast<int64_t>(ubBytes - reserve) / (4 * bytes), align);
    if (capacity <= 0) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalNum = total;
    tiling->blockFactor = block;
    tiling->ubFactor = block < capacity ? block : capacity;
    context->SetBlockDim(static_cast<uint32_t>(Ops::Base::CeilDiv(total, block)));
    workspace[0] = 0;
    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAbs([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AbsCompileInfo {};

IMPL_OP_OPTILING(Abs).Tiling(AbsTilingFunc).TilingParse<AbsCompileInfo>(TilingParseForAbs);

} // namespace optiling
