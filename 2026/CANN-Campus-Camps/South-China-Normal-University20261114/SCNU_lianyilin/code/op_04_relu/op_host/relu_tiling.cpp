/*!
 * \file relu_tiling.cpp
 * \brief CANNJudge 2980 模板对应的 Relu Tiling
 */
#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr int64_t ALIGN_BYTES = 32;
constexpr uint64_t RESERVED_UB_BYTES = 8 * 1024;
constexpr uint64_t TARGET_TILE_BYTES = 32 * 1024;
constexpr uint64_t QUEUE_BUFFER_COUNT = 4; // 输入、输出各两个 buffer

static inline int64_t CeilDivValue(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* platformInfo = context->GetPlatformInfo();
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    if (inputDesc == nullptr || inputShape == nullptr ||
        platformInfo == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const int64_t typeBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignElements = ALIGN_BYTES / typeBytes;

    // 必须匹配本模板 relu.cpp 中的 schMode：0=half，1=float。
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }

    int64_t totalNum = 1;
    const auto& shape = inputShape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return ge::GRAPH_FAILED;
        }
        if (dim == 0) {
            totalNum = 0;
            break;
        }
        if (totalNum > (std::numeric_limits<int64_t>::max() - ALIGN_BYTES) / dim) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= dim;
    }

    platform_ascendc::PlatformAscendC platform(platformInfo);
    int64_t coreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (coreNum <= 0 || ubBytes < RESERVED_UB_BYTES + QUEUE_BUFFER_COUNT * ALIGN_BYTES) {
        return ge::GRAPH_FAILED;
    }

    // 小输入减少启动核数，大输入使用可用向量核。
    if (totalNum <= 16384 && coreNum > 8) {
        coreNum = 8;
    }
    int64_t blockFactor = alignElements;
    int64_t blockDim = 1;
    if (totalNum > 0) {
        blockFactor = CeilDivValue(CeilDivValue(totalNum, coreNum), alignElements) * alignElements;
        blockDim = CeilDivValue(totalNum, blockFactor);
    }

    uint64_t tileBytes = (ubBytes - RESERVED_UB_BYTES) / QUEUE_BUFFER_COUNT;
    if (tileBytes > TARGET_TILE_BYTES) {
        tileBytes = TARGET_TILE_BYTES;
    }
    tileBytes = tileBytes / ALIGN_BYTES * ALIGN_BYTES;
    int64_t ubFactor = static_cast<int64_t>(tileBytes / typeBytes);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    // 沿用空工程原始结构体，不能换成 totalLength/tileNum。
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
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

}  // namespace optiling
