/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

// 小 shape 每核至少 4096 元素；16K shape 使用 4 核，减少调度开销。
constexpr uint32_t MIN_ELEMENTS_PER_CORE = 4096;
// 单核起始地址与常规 tile 均按 512 元素对齐，兼顾 910B 的 512B 带宽粒度。
constexpr uint32_t CORE_ALIGN_ELEMENTS = 512;
// FP16 双缓冲下 23040 元素占 180KB UB，可覆盖 40 核场景的单块
// 23040 元素；超过该值接近 UB 上限，实测容易触发调度超时。
constexpr uint32_t MAX_TILE_ELEMENTS = 23040;
constexpr uint32_t UB_RESERVED_BYTES = 2048;
// 小 shape 的 arch22 直通路径只占一份输入 UB，与通用双缓冲预算不同。
constexpr uint32_t LARGE_DISPATCH_THRESHOLD = 65536;
// 大 shape 限制为 20 核时，[45, 20480] 恰好每核两块 23040 元素，
// 单核可用双缓冲重叠相邻 tile 的搬运与计算。
constexpr uint32_t MAX_LARGE_BLOCK_DIM = 20;
constexpr uint32_t MAX_TOTAL_ELEMENTS = std::numeric_limits<uint32_t>::max();

static inline uint32_t CeilDivU32(uint64_t value, uint64_t divisor)
{
    return static_cast<uint32_t>((value + divisor - 1U) / divisor);
}

static inline uint32_t FloorAlignU32(uint64_t value, uint64_t align)
{
    return static_cast<uint32_t>(value / align * align);
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，由模板参数选择 FP32/FP16 kernel 实例。
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType inputDtype = inputDesc->GetDataType();
    if (inputDtype != ge::DT_FLOAT && inputDtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    uint32_t dtypeParam = static_cast<uint32_t>(inputDtype);
    ASCENDC_TPL_SEL_PARAM(context, dtypeParam);

    // 输入 shape 逐维相乘得到总元素数。
    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = 1;
    const gert::Shape& storageShape = xShape->GetStorageShape();
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        const int64_t dim = storageShape.GetDim(i);
        if (dim < 0) {
            return ge::GRAPH_FAILED;
        }
        if (dim == 0) {
            totalLength = 0;
            break;
        }
        if (static_cast<uint64_t>(dim) > MAX_TOTAL_ELEMENTS / totalLength) {
            return ge::GRAPH_FAILED;
        }
        totalLength *= static_cast<uint32_t>(dim);
    }
    if (storageShape.GetDimNum() == 0) {
        totalLength = 1;
    }

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = totalLength;

    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t platformAivNum = ascendcPlatform.GetCoreNumAiv();
    if (platformAivNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t aivNum = static_cast<uint32_t>(platformAivNum);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize <= UB_RESERVED_BYTES) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t elemBytes = (inputDtype == ge::DT_FLOAT16) ? 2U : 4U;
    const uint32_t alignElements = 32U / elemBytes;
    const uint32_t queueBuffers = (totalLength < LARGE_DISPATCH_THRESHOLD) ? 1U : 2U;
    const uint32_t bytesPerElement = 2U * queueBuffers * elemBytes;
    uint32_t ubTileElements = FloorAlignU32(
        (ubSize - UB_RESERVED_BYTES) / bytesPerElement, CORE_ALIGN_ELEMENTS);
    ubTileElements = std::min(ubTileElements, MAX_TILE_ELEMENTS);
    if (ubTileElements < alignElements) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = 1;
    uint32_t blockLength = CORE_ALIGN_ELEMENTS;
    uint32_t tileLength = CORE_ALIGN_ELEMENTS;
    if (totalLength > 0) {
        blockDim = CeilDivU32(totalLength, MIN_ELEMENTS_PER_CORE);
        blockDim = std::max(1U, std::min(blockDim, aivNum));
        if (totalLength >= LARGE_DISPATCH_THRESHOLD) {
            blockDim = std::min(blockDim, MAX_LARGE_BLOCK_DIM);
        }

        const uint32_t perCore = CeilDivU32(totalLength, blockDim);
        const uint64_t alignedBlockLength =
            (static_cast<uint64_t>(perCore) + CORE_ALIGN_ELEMENTS - 1U) /
            CORE_ALIGN_ELEMENTS * CORE_ALIGN_ELEMENTS;
        if (alignedBlockLength > MAX_TOTAL_ELEMENTS) {
            blockLength = totalLength;
        } else {
            blockLength = static_cast<uint32_t>(alignedBlockLength);
        }
        blockLength = std::max(blockLength, alignElements);

        blockDim = CeilDivU32(totalLength, blockLength);
        blockDim = std::max(1U, std::min(blockDim, aivNum));

        tileLength = std::min(blockLength, ubTileElements);
        tileLength = FloorAlignU32(tileLength, CORE_ALIGN_ELEMENTS);
        if (tileLength < alignElements) {
            tileLength = alignElements;
        }
    }

    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    context->SetBlockDim(blockDim);

    // ReLU 不产生中间 workspace。
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
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
