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

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_DATA_NUMBER = 3;   // input + output + 余量（避免 UB 满载）
constexpr uint32_t SMALL_SHAPE_MAX = 1000000;

static void CalcSmallTiling(gert::TilingContext* context, ReluTilingData* tiling,
                            uint32_t dataSz, uint32_t coreNum, uint32_t typeLength,
                            uint64_t ubSize)
{
    uint32_t inputLength = dataSz * typeLength;
    uint32_t inputLengthAlign32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;
    if (coreNum > totalBlockNum) {
        coreNum = std::max(totalBlockNum, 1u);
    }
    context->SetBlockDim(coreNum);

    uint32_t everyCoreBlockNum = totalBlockNum / coreNum;
    uint32_t tailBlockNum = totalBlockNum % coreNum;

    uint32_t tileBlockNum = (static_cast<uint32_t>(ubSize) / BLOCK_SIZE / BUFFER_NUM) / UB_DATA_NUMBER;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

    uint32_t smallCoreDataNum = everyCoreBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
}

static void CalcBigTiling(gert::TilingContext* context, ReluTilingData* tiling,
                          uint32_t dataSz, uint32_t coreNum, uint32_t typeLength)
{
    // 取能整除 data_sz 的最大核数（大 shape 用满核并行）
    while (coreNum > 1 && (dataSz % coreNum) != 0) {
        coreNum--;
    }
    context->SetBlockDim(coreNum);

    uint32_t blockLength = dataSz / coreNum;
    uint32_t alignNum = BLOCK_SIZE / typeLength;
    uint32_t tileLength = 8192;  // 每循环元素数（float 32KB）
    if (blockLength < tileLength) {
        tileLength = blockLength;
    }
    tileLength = (tileLength / alignNum) * alignNum;
    if (tileLength == 0) {
        tileLength = alignNum;
    }
    uint32_t tileNum = blockLength / (tileLength * BUFFER_NUM);
    if (tileNum < 1) {
        tileNum = 1;
        tileLength = blockLength;
    }

    tiling->blockLength = blockLength;
    tiling->tileNum = tileNum;
    tiling->tileLength = tileLength;
    tiling->smallCoreDataNum = 0;
    tiling->bigCoreDataNum = 0;
    tiling->finalBigTileNum = 0;
    tiling->finalSmallTileNum = 0;
    tiling->tileDataNum = tileLength;
    tiling->smallTailDataNum = 0;
    tiling->bigTailDataNum = 0;
    tiling->tailBlockNum = 0;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = data_sz;

    // 平台信息
    uint64_t ubSize = 0;
    uint32_t platformCoreNum = 40;
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        int64_t coreNumAiv = ascendcPlatform.GetCoreNumAiv();
        if (coreNumAiv > 0) {
            platformCoreNum = static_cast<uint32_t>(coreNumAiv);
        }
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    if (ubSize == 0) {
        ubSize = 192 * 1024;
    }

    uint32_t typeLength = (inputDesc != nullptr &&
                           (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) ? 2 : 4;

    // 大小 shape 分离：小 shape 用官方大核/小核（4核），大 shape 用 32 核整除流水
    if (static_cast<uint32_t>(data_sz) <= SMALL_SHAPE_MAX) {
        CalcSmallTiling(context, tiling, static_cast<uint32_t>(data_sz), 4, typeLength, ubSize);
    } else {
        CalcBigTiling(context, tiling, static_cast<uint32_t>(data_sz), 16, typeLength);
    }

    // 配置 workspace 大小
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
