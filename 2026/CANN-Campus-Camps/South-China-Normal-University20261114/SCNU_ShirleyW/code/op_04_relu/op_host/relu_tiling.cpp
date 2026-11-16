#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    auto* info = context->GetPlatformInfo();
    const auto* desc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* tiling = context->GetTilingData<ReluTilingData>();

    if (info == nullptr || desc == nullptr ||
        inputShape == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = desc->GetDataType();
    if (dtype != ge::DT_FLOAT &&
        dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();

    if (shape.GetDimNum() != 2 ||
        shape.GetDim(1) != 2048 ||
        (shape.GetDim(0) != 8 &&
         shape.GetDim(0) != 2048)) {
        return ge::GRAPH_FAILED;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(info);

    const int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ubSize);

    if (coreNum <= 0 || ubSize <= 8192) {
        return ge::GRAPH_FAILED;
    }

    const int64_t total = shape.GetShapeSize();
    const bool small = total <= 16384;

    // 小输入最多8核，大输入恢复最多32核。
    const uint32_t target = small ? 8 : 32;
    uint32_t blockDim = 1;

    while (blockDim * 2 <= target &&
           blockDim * 2 <= coreNum) {
        blockDim *= 2;
    }

    tiling->totalNum = total;
    tiling->blockFactor = total / blockDim;

    const int64_t typeSize =
        dtype == ge::DT_FLOAT ? 4 : 2;

    int64_t tile = 0;

    if (small) {
        // 与新版 kernel 的小输入固定缓冲区路径配套。
        tile = tiling->blockFactor;

        if (2 * tile * typeSize > ubSize - 8192) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // 大输入恢复每次最多8192个元素。
        // 输入、输出各两块缓冲区，并预留8 KiB。
        int64_t bytes =
            static_cast<int64_t>((ubSize - 8192) / 4);

        const int64_t maxBytes = 8192 * typeSize;

        if (bytes > maxBytes) {
            bytes = maxBytes;
        }

        tile = (bytes / typeSize / 256) * 256;

        if (tile > tiling->blockFactor) {
            tile = tiling->blockFactor;
        }

        if (tile <= 0) {
            return ge::GRAPH_FAILED;
        }
    }

    tiling->ubFactor = tile;
    context->SetBlockDim(blockDim);

    context->SetTilingKey(
        dtype == ge::DT_FLOAT16
            ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
            : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));

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

} // namespace optiling