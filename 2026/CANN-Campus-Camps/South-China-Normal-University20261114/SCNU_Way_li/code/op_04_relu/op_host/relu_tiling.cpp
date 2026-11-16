/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* tiling = context->GetTilingData<ReluTilingData>();
    auto* platformInfo = context->GetPlatformInfo();

    if (inputDesc == nullptr || inputShape == nullptr ||
        tiling == nullptr || platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 支持 float32、float16。
    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    // 对应赛题中的两种输入形状。
    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 2048 ||
        (shape.GetDim(0) != 8 && shape.GetDim(0) != 2048)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength =
        static_cast<uint32_t>(shape.GetDim(0) * shape.GetDim(1));

    const uint32_t typeBytes =
        (dtype == ge::DT_FLOAT) ? 4U : 2U;

    // 查询可用 Vector 核数和 UB 大小。
    platform_ascendc::PlatformAscendC platform(platformInfo);

    const uint32_t availableCores = platform.GetCoreNumAiv();

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ubBytes);

    if (availableCores == 0 || ubBytes < 1024) {
        return ge::GRAPH_FAILED;
    }

    // 两个输入的元素总数均为 2 的幂。
    // 核数也选 2 的幂，以保证均匀分核，没有尾块。
    // 每核至少处理 2048 个元素。
    const uint32_t coreLimit = totalLength / 2048U;
    uint32_t coreNum = 1;

    while (coreNum * 2U <= availableCores &&
           coreNum * 2U <= coreLimit) {
        coreNum *= 2U;
    }

    const uint32_t blockLength = totalLength / coreNum;

    // 双缓冲：输入 2 块、输出 2 块，共 4 块。
    // 每块最大 32 KiB，并受实际 UB 大小限制。
    uint64_t tileBytes = ubBytes / 4U;

    if (tileBytes > 32U * 1024U) {
        tileBytes = 32U * 1024U;
    }

    tileBytes = tileBytes / 256U * 256U;

    // 与模板保持一致：
    // tileLength = blockLength / tileNum / BUFFER_NUM
    // BUFFER_NUM = 2。
    uint32_t tileNum = 1;

    while (static_cast<uint64_t>(
               blockLength / tileNum / 2U) * typeBytes > tileBytes) {
        tileNum *= 2U;
    }

  // 与 relu.cpp 的入口约定一致：
// schMode = 0 -> half
// schMode = 1 -> float
uint32_t schMode = (dtype == ge::DT_FLOAT16)
    ? RELU_TPL_SCH_MODE_0
    : RELU_TPL_SCH_MODE_1;

ASCENDC_TPL_SEL_PARAM(context, schMode);

    tiling->totalLength = totalLength;
    tiling->tileNum = tileNum;

    context->SetBlockDim(coreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling