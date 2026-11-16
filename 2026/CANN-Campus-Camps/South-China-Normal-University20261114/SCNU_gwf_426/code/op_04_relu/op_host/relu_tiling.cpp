#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) return ge::GRAPH_FAILED;
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    if (x1_shape == nullptr) return ge::GRAPH_FAILED;

    int64_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    if (data_sz <= 0) return ge::GRAPH_FAILED;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) coreNum = 1;
    if (data_sz < coreNum) coreNum = 1;
    context->SetBlockDim(coreNum);

    uint32_t blockLength = static_cast<uint32_t>((data_sz + coreNum - 1) / coreNum);
    uint32_t lastBlockLength = static_cast<uint32_t>(data_sz - (coreNum - 1) * blockLength);

    uint32_t dtypeSize = (DT_X == ge::DT_FLOAT16) ? 2 : 4;
    uint32_t alignNum = BLOCK_SIZE / dtypeSize;

    // ascend910b UB 可按 192KB 估算，in+out 双缓冲各占一份
    constexpr uint32_t UB_BYTES = 192 * 1024;
    uint32_t maxTileElems = UB_BYTES / (BUFFER_NUM * dtypeSize * 2);
    maxTileElems = (maxTileElems / alignNum) * alignNum;
    if (maxTileElems == 0) maxTileElems = alignNum;

    if (maxTileElems > blockLength) {
        maxTileElems = (blockLength / alignNum) * alignNum;
        if (maxTileElems == 0) maxTileElems = alignNum;
    }

    uint32_t tileLength = maxTileElems;
    uint32_t tileNum = blockLength / tileLength;
    uint32_t lastTileLength = blockLength - tileNum * tileLength;

    tiling->totalLength = static_cast<uint32_t>(data_sz);
    tiling->blockLength = blockLength;
    tiling->lastBlockLength = lastBlockLength;
    tiling->tileNum = tileNum;
    tiling->tileLength = tileLength;
    tiling->lastTileLength = lastTileLength;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
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