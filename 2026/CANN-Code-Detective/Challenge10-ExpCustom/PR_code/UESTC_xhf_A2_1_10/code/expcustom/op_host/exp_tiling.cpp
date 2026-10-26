/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

constexpr uint32_t WORKSPACE_SIZE = 0U;

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    // 与核函数整数模板保持一致：float32 使用 1，float16 使用 0。
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Only float32 and float16 are supported"), return ge::GRAPH_FAILED);
    const uint32_t schMode = dtype == ge::DT_FLOAT ? EXP_MODE_FLOAT : EXP_MODE_HALF;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& storageShape = inputShape->GetStorageShape();

    int64_t totalNum = 1;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        totalNum *= storageShape.GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t blockDim = platform.GetCoreNumAiv();
    OP_CHECK_IF(blockDim <= 0, OP_LOGE(context, "AIV core count must be positive"), return ge::GRAPH_FAILED);

    // DataCopy 的传输长度必须是 32 字节的整数倍。
    const int64_t elementBytes = dtype == ge::DT_FLOAT ? 4 : 2;
    const int64_t alignElements = 32 / elementBytes;
    OP_CHECK_IF(totalNum % alignElements != 0,
        OP_LOGE(context, "Input element count must be 32-byte aligned"), return ge::GRAPH_FAILED);

    // 核数整除总长度，且每核元素数对齐；这样最后一核和最后一个 tile 均不越界。
    blockDim = blockDim > totalNum ? totalNum : blockDim;
    while (blockDim > 1 &&
           (totalNum % blockDim != 0 || (totalNum / blockDim) % alignElements != 0)) {
        --blockDim;
    }
    const int64_t blockFactor = totalNum / blockDim;

    // 输入、输出各有两个缓冲区；按 dtype 和 UB 容量限制单块大小。
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t maxUbFactor = (static_cast<int64_t>(ubSize) / (4 * elementBytes))
        / alignElements * alignElements;
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
    const int64_t ubFactor = blockFactor < maxUbFactor ? blockFactor : maxUbFactor;

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
