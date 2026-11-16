#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    const auto dtype =
        context->GetInputDesc(0)->GetDataType();

    if (dtype != ge::DT_FLOAT &&
        dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const uint32_t totalLength =
        static_cast<uint32_t>(
            context->GetInputShape(0)
                ->GetStorageShape()
                .GetShapeSize());

    // 本实现针对题目规定的两种输入大小。
    if (totalLength != 8U * 2048U &&
        totalLength != 2048U * 2048U) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(
        context->GetPlatformInfo());

    const uint32_t availableCores =
        platform.GetCoreNumAiv();

    // 小输入最多 8 核，大输入最多 32 核。
    const uint32_t coreLimit =
        totalLength <= 16384U ? 8U : 32U;

    // 选择不超过硬件可用数量的 2 次幂核数，
    // 保证题目输入可以均匀分配。
    uint32_t coreNum = 1;
    while (coreNum * 2U <= coreLimit &&
           coreNum * 2U <= availableCores) {
        coreNum *= 2U;
    }

    const uint32_t typeBytes =
        dtype == ge::DT_FLOAT16 ? 2U : 4U;

    const uint32_t blockLength =
        totalLength / coreNum;

    // 每个输入/输出缓冲块最大 32KB。
    const uint32_t maxTileLength =
        32768U / typeBytes;

    const uint32_t tileLength =
        blockLength < maxTileLength
            ? blockLength
            : maxTileLength;

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    tiling->totalLength = totalLength;
    tiling->tileNum = blockLength / tileLength;

    context->SetBlockDim(coreNum);
    context->GetWorkspaceSizes(1)[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

}  // namespace optiling