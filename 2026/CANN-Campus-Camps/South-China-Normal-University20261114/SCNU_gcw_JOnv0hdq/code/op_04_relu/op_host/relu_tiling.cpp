/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

#include <algorithm>
#include <limits>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr int64_t COPY_ALIGN_BYTES = 32;
constexpr int64_t BLOCK_ALIGN_ELEMENTS = 512;
constexpr int64_t MIN_BYTES_PER_CORE = 4096;
constexpr int64_t SMALL_INPUT_MAX_BYTES = 64 * 1024;
constexpr int64_t SMALL_INPUT_BYTES_PER_CORE = 8192;
constexpr int64_t TILE_ALIGN_BYTES = 512;
constexpr int64_t MAX_VECTOR_BYTES = 255 * 256;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Relu expects float32 or float16"), return ge::GRAPH_FAILED);
    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const int64_t elemBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t totalLength = inputShape->GetStorageShape().GetShapeSize();
    const int64_t copyAlign = COPY_ALIGN_BYTES / elemBytes;
    OP_CHECK_IF(totalLength <= 0 || totalLength > std::numeric_limits<uint32_t>::max()
        || totalLength % copyAlign != 0,
        OP_LOGE(context, "Relu expects a nonempty 32-byte-aligned input"), return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t availableCores = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(availableCores <= 0 || ubSize == 0,
        OP_LOGE(context, "Invalid vector core count or UB capacity"), return ge::GRAPH_FAILED);

    // 小输入尝试用较少的核，前提是单缓冲能容纳每核的目标数据量。
    const int64_t singleBufferTileBytes = FloorAlign(
        std::min(static_cast<int64_t>(ubSize) / 2, MAX_VECTOR_BYTES), TILE_ALIGN_BYTES);
    const int64_t inputBytes = totalLength * elemBytes;
    const int64_t minBytesPerCore =
        inputBytes <= SMALL_INPUT_MAX_BYTES && singleBufferTileBytes >= SMALL_INPUT_BYTES_PER_CORE
        ? SMALL_INPUT_BYTES_PER_CORE : MIN_BYTES_PER_CORE;

    // 按 512 元素为一组分核，Kernel 将多出的组分给前几个核。
    const int64_t blockGroups = CeilDiv(totalLength, BLOCK_ALIGN_ELEMENTS);
    const int64_t desiredCores = std::max<int64_t>(1, inputBytes / minBytesPerCore);
    const int64_t usedCores = std::min(availableCores, desiredCores);
    const int64_t maxBlockLength = std::min(totalLength, CeilDiv(blockGroups, usedCores) * BLOCK_ALIGN_ELEMENTS);
    const int64_t maxBlockBytes = maxBlockLength * elemBytes;

    // 一批能装下时，输入、输出各用一个缓冲。
    int64_t tileNum = 1;
    if (maxBlockBytes > singleBufferTileBytes) {
        // 大输入只给输入分配一个缓冲，给输出分配两个，以增大每次搬运量。
        const int64_t bufferCount = inputBytes > SMALL_INPUT_MAX_BYTES ? 3 : 4;
        const int64_t maxTileBytes = FloorAlign(
            std::min(static_cast<int64_t>(ubSize) / bufferCount, MAX_VECTOR_BYTES), TILE_ALIGN_BYTES);
        OP_CHECK_IF(maxTileBytes == 0,
            OP_LOGE(context, "UB is too small for Relu buffering"), return ge::GRAPH_FAILED);
        tileNum = CeilDiv(maxBlockBytes, maxTileBytes);
    }

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->tileNum = static_cast<uint32_t>(tileNum);
    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
