/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */
#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {
namespace {
constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t CACHE_LINE_BLOCKS = 16U;     // 512B L2 Cache Line = 16 * 32B
constexpr uint32_t BUFFER_NUM = 2U;             // 硬件原生 Ping-Pong 双缓冲
constexpr uint32_t QUEUE_NUM = 2U;
constexpr uint64_t SMALL_BYTES_LIMIT = 65536ULL; // 64KB 以内走极低启动开销路径
constexpr uint64_t DEFAULT_UB_SIZE = 196608ULL;  // Ascend 910B 192KB UB
} // namespace

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 根据输入 dtype 设置 tilingKey（float -> float，float16 -> float16）
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dtype = inputDesc->GetDataType();
    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取输入 shape 总元素数
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    uint32_t totalLength = 1U;
    const gert::Shape& storageShape = x1_shape->GetStorageShape();
    for (size_t i = 0; i < storageShape.GetDimNum(); i++) {
        totalLength *= static_cast<uint32_t>(storageShape.GetDim(i));
    }

    if (totalLength == 0U) {
        tiling->totalLength = 0U;
        tiling->tileNum = 0U;
        context->SetBlockDim(1U);
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0U) {
        aivNum = 48U;
    }

    uint64_t ubSize = 0ULL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0ULL) {
        ubSize = DEFAULT_UB_SIZE;
    }

    uint32_t dtypeBytes = (dtype == ge::DT_FLOAT16) ? 2U : 4U;
    uint32_t alignElems = BLOCK_SIZE / dtypeBytes;
    uint32_t totalBlocks = (totalLength + alignElems - 1U) / alignElems;
    uint64_t totalBytes = static_cast<uint64_t>(totalBlocks) * BLOCK_SIZE;

    // 1. 确定最优核数：64KB 以内用 1~2 核避免多核启动与同步开销；大张量打满全部 AIV 核
    uint32_t blockDim = 1U;
    if (totalBytes <= SMALL_BYTES_LIMIT) {
        blockDim = (totalBytes >= 32768ULL && aivNum >= 2U) ? 2U : 1U;
    } else {
        uint32_t maxUsefulCores = totalBlocks / CACHE_LINE_BLOCKS;
        if (maxUsefulCores == 0U) {
            maxUsefulCores = 1U;
        }
        blockDim = std::min(aivNum, maxUsefulCores);
    }
    context->SetBlockDim(blockDim);

    // 2. 按 512B L2 Cache Line (16 blocks) 均分多核，避免缓存行伪共享与尾核不均
    uint32_t is512Aligned = ((totalBlocks % CACHE_LINE_BLOCKS) == 0U) ? 1U : 0U;
    uint32_t unitBlocks = (is512Aligned != 0U) ? CACHE_LINE_BLOCKS : 1U;
    uint32_t totalUnits = totalBlocks / unitBlocks;

    uint32_t baseUnits = totalUnits / blockDim;
    uint32_t extraCount = totalUnits % blockDim;
    uint32_t maxCoreUnits = baseUnits + (extraCount > 0U ? 1U : 0U);

    // 3. 按 DoubleBuffer (2 * 2 = 4 份缓冲区) 计算 512B 对齐的最大单 Tile 大小
    uint64_t safeUb = (ubSize > 4096ULL) ? (ubSize - 4096ULL) : ubSize;
    uint32_t maxTileBlocks = static_cast<uint32_t>(safeUb / (QUEUE_NUM * BUFFER_NUM * BLOCK_SIZE));
    if (maxTileBlocks > CACHE_LINE_BLOCKS) {
        maxTileBlocks &= ~(CACHE_LINE_BLOCKS - 1U);
    }
    uint32_t maxTileUnits = std::max(1U, maxTileBlocks / unitBlocks);
    if (maxTileUnits > 0x3FFU) {
        maxTileUnits = 0x3FFU;
    }

    uint32_t loopNum = (maxCoreUnits + maxTileUnits - 1U) / maxTileUnits;
    if (loopNum == 0U) {
        loopNum = 1U;
    }
    uint32_t tileUnits = (maxCoreUnits + loopNum - 1U) / loopNum;
    if (tileUnits > maxTileUnits) {
        tileUnits = maxTileUnits;
    }

    // 位域打包：[31] is512Aligned | [30:21] tileUnits | [20:14] extraCount | [13:0] baseUnits
    tiling->totalLength = totalLength;
    tiling->tileNum = (baseUnits & 0x3FFFU) |
                      ((extraCount & 0x7FU) << 14U) |
                      ((tileUnits & 0x3FFU) << 21U) |
                      (is512Aligned << 31U);

    // 设置 workspace 大小为 0
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