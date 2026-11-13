#include "register/op_def_registry.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {

static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
{
    const auto* input = context->GetInputDesc(0);
    const auto* shape = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    auto* platformInfo = context->GetPlatformInfo();
    auto* tiling = context->GetTilingData<LeakyreluTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!input || !shape || !attrs || !platformInfo || !tiling || !workspace) {
        return ge::GRAPH_FAILED;
    }

    const float* slope = attrs->GetAttrPointer<float>(0);
    const auto dtype = input->GetDataType();
    if (!slope || (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t total = shape->GetStorageShape().GetShapeSize();
    const int64_t alignment = dtype == ge::DT_FLOAT16 ? 16 : 8;
    const int64_t aivCores = platform_ascendc::PlatformAscendC(platformInfo).GetCoreNumAiv();
    if (total <= 0 || total % alignment != 0 || aivCores <= 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t cores = aivCores < 8 ? aivCores : 8;
    using Ops::Base::CeilAlign;
    using Ops::Base::CeilDiv;
    tiling->totalNum = total;
    tiling->blockFactor = CeilAlign(CeilDiv(total, cores), alignment);
    tiling->ubFactor = 256;
    tiling->negativeSlope = *slope;
    context->SetBlockDim(CeilDiv(total, tiling->blockFactor));
    const uint32_t mode = dtype == ge::DT_FLOAT16 ? LEAKYRELU_TPL_SCH_MODE_0 : LEAKYRELU_TPL_SCH_MODE_1;
    context->SetTilingKey(GET_TPL_TILING_KEY(mode));
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
