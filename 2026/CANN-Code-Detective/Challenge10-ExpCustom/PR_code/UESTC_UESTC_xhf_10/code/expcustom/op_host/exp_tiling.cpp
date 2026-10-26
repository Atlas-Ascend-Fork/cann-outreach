/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include <limits>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* tiling = context->GetTilingData<ExpTilingData>();
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    int64_t totalNum = 1;
    const auto& shape = inputShape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "negative dimension"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(dim > 0 && totalNum > std::numeric_limits<int64_t>::max() / dim,
            OP_LOGE(context, "input size overflow"), return ge::GRAPH_FAILED);
        totalNum *= dim;
    }
    // 标量含一个元素；零元素张量启动一个立即返回的空任务。
    if (shape.GetDimNum() == 0) {
        totalNum = 1;
    }

    const auto dtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = sizeof(float);
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = sizeof(uint16_t);
    } else {
        OP_LOGE(context, "Exp only supports float32 and float16");
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(platformInfo);
    const int64_t maxCoreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(maxCoreNum <= 0 || ubSize == 0,
        OP_LOGE(context, "invalid platform resource"), return ge::GRAPH_FAILED);

    const int64_t alignNum = 32 / typeSize;
    const int64_t blockFactor = totalNum == 0 ? alignNum : CeilAlign(CeilDiv(totalNum, maxCoreNum), alignNum);
    const int64_t usedCoreNum = totalNum == 0 ? 1 : CeilDiv(totalNum, blockFactor);
    // 输入、输出各使用双缓冲，因此每个元素预留 4 份存储空间。
    int64_t ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / (4 * typeSize), alignNum);
    ubFactor = ubFactor > blockFactor ? blockFactor : ubFactor;
    ubFactor = ubFactor < alignNum ? alignNum : ubFactor;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    const uint32_t schMode = dtype == ge::DT_FLOAT ? EXP_TPL_SCH_MODE_1 : EXP_TPL_SCH_MODE_0;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
