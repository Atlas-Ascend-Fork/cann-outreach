/*!
 * \file abs_tiling.cpp
 * \brief Abs 算子 Tiling 实现
 */
#include "register/op_def_registry.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/abs_tiling_data.h"
#include "../op_kernel/abs_tiling_key.h"

namespace optiling {
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

static ge::graphStatus AbsTilingFunc(gert::TilingContext* context)
{
    const auto* input = context->GetInputDesc(0);
    const auto* shape = context->GetInputShape(0);
    auto* platformInfo = context->GetPlatformInfo();
    auto* tiling = context->GetTilingData<AbsTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!input || !shape || !platformInfo || !tiling || !workspace) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = input->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const bool isHalf = dtype == ge::DT_FLOAT16;
    constexpr int64_t dmaAlignment = 32;
    constexpr int64_t queueBuffers = 4;
    constexpr int64_t maxTileBytes = 8192;
    constexpr int64_t targetCoreElements = 512;
    const int64_t elementBytes = isHalf ? 2 : 4;
    const int64_t alignElements = dmaAlignment / elementBytes;

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t availableCores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t availableTileBytes = static_cast<int64_t>(ubBytes / queueBuffers);
    const int64_t tileBytes = availableTileBytes < maxTileBytes ? availableTileBytes : maxTileBytes;
    const int64_t maxTileElements = FloorAlign(tileBytes, dmaAlignment) / elementBytes;
    const int64_t total = shape->GetStorageShape().GetShapeSize();
    if (availableCores <= 0 || maxTileElements == 0 || total < 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t nonzeroTotal = total == 0 ? 1 : total;
    const int64_t desiredCores = CeilDiv(nonzeroTotal, targetCoreElements);
    const int64_t cores = desiredCores < availableCores ? desiredCores : availableCores;
    const int64_t block = CeilAlign(CeilDiv(nonzeroTotal, cores), alignElements);
    tiling->totalNum = total;
    tiling->blockFactor = block;
    tiling->ubFactor = block < maxTileElements ? block : maxTileElements;
    context->SetBlockDim(CeilDiv(nonzeroTotal, block));
    const uint32_t mode = isHalf ? ABS_TPL_SCH_MODE_0 : ABS_TPL_SCH_MODE_1;
    context->SetTilingKey(GET_TPL_TILING_KEY(mode));
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAbs([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AbsCompileInfo {};

IMPL_OP_OPTILING(Abs).Tiling(AbsTilingFunc).TilingParse<AbsCompileInfo>(TilingParseForAbs);
} // namespace optiling
