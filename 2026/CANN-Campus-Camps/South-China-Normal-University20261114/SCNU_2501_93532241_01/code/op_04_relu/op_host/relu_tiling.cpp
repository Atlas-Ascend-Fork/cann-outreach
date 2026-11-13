/*!
 * \file relu_tiling.cpp
 * \brief ReLU tiling: keep the supplied totalNum/blockFactor/ubFactor ABI.
 */

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {
namespace {
constexpr int64_t ALIGN_BYTES = 512;
constexpr int64_t SMALL_INPUT_BYTES = 64 * 1024;
constexpr int64_t SMALL_CORE_NUM = 8;
constexpr int64_t TARGET_TILE_BYTES = 32 * 1024;
constexpr uint64_t UB_RESERVE_BYTES = 8 * 1024;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return coreNum > 0 && ubSize >= UB_RESERVE_BYTES + 4 * ALIGN_BYTES
        ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* tiling = context->GetTilingData<ReluTilingData>();
    if (inputDesc == nullptr || inputShape == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const int64_t elementBytes = dtype == ge::DT_FLOAT ? 4 : 2;
    const int64_t alignElements = ALIGN_BYTES / elementBytes;
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    // All four shape/dtype combinations in this problem are 512-byte aligned.
    if (totalNum <= 0 || totalNum % alignElements != 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (totalNum <= SMALL_INPUT_BYTES / elementBytes) {
        coreNum = std::min(coreNum, SMALL_CORE_NUM);
    }

    const int64_t totalBlocks = totalNum / alignElements;
    coreNum = std::min(coreNum, totalBlocks);
    const int64_t blocksPerCore = totalBlocks / coreNum + (totalBlocks % coreNum != 0);
    const int64_t blockFactor = blocksPerCore * alignElements;
    const int64_t usedCores = totalNum / blockFactor + (totalNum % blockFactor != 0);

    // Two input buffers and two output buffers. The kernel uses one each
    // when a core needs only one tile.
    const uint64_t maxTileBytes = std::min<uint64_t>(
        TARGET_TILE_BYTES, ((ubSize - UB_RESERVE_BYTES) / 4 / ALIGN_BYTES) * ALIGN_BYTES);
    const int64_t ubFactor = std::min(blockFactor, static_cast<int64_t>(maxTileBytes / elementBytes));

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    // These are schMode values from the supplied relu.cpp, NOT ge::DataType values.
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
