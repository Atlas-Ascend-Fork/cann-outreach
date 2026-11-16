
/*!
 * \file relu_tiling.cpp
 * \brief ReLU 第五轮性能优化
 *
 * 小数据：单核优先，降低启动调度开销
 * 大数据：使用全部可用 AIV 核
 * Tile：64KB
 * Buffer：单缓冲，降低 UB 占用
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <climits>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

namespace {

constexpr uint32_t WS_SYS_SIZE = 0U;

constexpr int64_t BLOCK_BYTES = 32;

constexpr int64_t TYPE_SIZE_FP32 = 4;
constexpr int64_t TYPE_SIZE_FP16 = 2;

// 单缓冲
constexpr int64_t UB_BUFFER_NUM = 2;

// Tile 目标大小：64KB
constexpr int64_t TILE_BYTES_TARGET = 32 * 1024;

// GM 对齐
constexpr int64_t GM_ALIGN_BYTES = 512;

// 大数据每核最少元素数
constexpr int64_t MIN_ELEMS_PER_CORE = 1024;

// UB 预留空间
constexpr int64_t UB_RESERVED_BYTES = 2 * 1024;

// 小数据判断阈值
constexpr int64_t SMALL_DATA_THRESHOLD = 32768;

} // namespace

// 获取设备信息
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context, platformInfoPtr);

    platform_ascendc::PlatformAscendC ascendcPlatform(
        platformInfoPtr);

    coreNum = static_cast<int64_t>(
        ascendcPlatform.GetCoreNumAiv());

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context, "Invalid AIV core number"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "Invalid UB size"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 配置 Workspace
static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context, currentWorkspace);

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

// ReLU Tiling
static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // 1. 获取数据类型
    const auto* inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context, inputDesc);

    const ge::DataType dataType =
        inputDesc->GetDataType();

    int64_t typeSize = 0;
    uint32_t schMode = 0;

    if (dataType == ge::DT_FLOAT16) {
        typeSize = TYPE_SIZE_FP16;
        schMode = RELU_TPL_SCH_MODE_0;
    } else if (dataType == ge::DT_FLOAT) {
        typeSize = TYPE_SIZE_FP32;
        schMode = RELU_TPL_SCH_MODE_1;
    } else {
        OP_LOGE(context, "Unsupported ReLU data type");
        return ge::GRAPH_FAILED;
    }

    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 2. 获取设备信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    // 3. 配置 Workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);

    // 4. 获取 TilingData
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context, tiling);

    // 5. 获取 Shape
    const gert::StorageShape* xShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context, xShape);

    const gert::Shape& storageShape =
        xShape->GetStorageShape();

    // 6. 计算总元素数量
    int64_t totalLength = 1;

    for (size_t i = 0;
         i < static_cast<size_t>(storageShape.GetDimNum());
         ++i) {

        const int64_t dim =
            storageShape.GetDim(i);

        OP_CHECK_IF(
            dim <= 0 || totalLength > INT64_MAX / dim,
            OP_LOGE(context, "Invalid input shape"),
            return ge::GRAPH_FAILED);

        totalLength *= dim;
    }

    OP_CHECK_IF(
        totalLength <= 0 || totalLength > UINT32_MAX,
        OP_LOGE(context, "Invalid totalLength"),
        return ge::GRAPH_FAILED);

    // 7. 计算对齐元素数量
    const int64_t alignElems =
        BLOCK_BYTES / typeSize;

    const int64_t gmAlignElems =
        GM_ALIGN_BYTES / typeSize;

    // 8. 选择核数：保持第四轮策略
    int64_t usedCoreNum = 1;

    if (totalLength <= SMALL_DATA_THRESHOLD) {
        // 小数据：单核
        usedCoreNum = 1;
    } else {
        // 大数据：尽可能使用全部 AIV 核
        const int64_t coreLimit =
            CeilDiv(totalLength, MIN_ELEMS_PER_CORE);

        usedCoreNum = std::min(coreNum, coreLimit);
    }

    usedCoreNum =
        std::max<int64_t>(usedCoreNum, 1);

    // 9. 计算每核处理长度
    int64_t blockLength = CeilAlign(
        CeilDiv(totalLength, usedCoreNum),
        gmAlignElems);

    // 修正实际使用核数
    usedCoreNum = std::min<int64_t>(
        usedCoreNum,
        CeilDiv(totalLength, blockLength));

    usedCoreNum =
        std::max<int64_t>(usedCoreNum, 1);

    // 重新计算每核长度
    blockLength = CeilAlign(
        CeilDiv(totalLength, usedCoreNum),
        gmAlignElems);

    // 10. 计算 UB 可用空间
    OP_CHECK_IF(
        ubSize <= static_cast<uint64_t>(UB_RESERVED_BYTES),
        OP_LOGE(context, "UB size is insufficient"),
        return ge::GRAPH_FAILED);

    const uint64_t usableUb =
        ubSize - static_cast<uint64_t>(UB_RESERVED_BYTES);

    // 输入和输出各使用一个 Buffer
    const uint64_t bufferCount =
        2ULL * UB_BUFFER_NUM;

    int64_t maxTileElems =
        static_cast<int64_t>(
            usableUb /
            (bufferCount * static_cast<uint64_t>(typeSize)));

    maxTileElems = FloorAlign(
        maxTileElems,
        gmAlignElems);

    // 11. Tile 长度：目标 64KB
    int64_t tileLength = FloorAlign(
        TILE_BYTES_TARGET / typeSize,
        gmAlignElems);

    // 根据 UB 容量限制实际 Tile 长度
    tileLength = std::min(
        tileLength,
        maxTileElems);

    OP_CHECK_IF(
        tileLength < alignElems,
        OP_LOGE(context, "Invalid tileLength"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tileLength > UINT32_MAX ||
        blockLength > UINT32_MAX,
        OP_LOGE(context, "Tiling parameter overflow"),
        return ge::GRAPH_FAILED);

    // 12. 计算 Tile 数量
    int64_t tileNum =
        CeilDiv(blockLength, tileLength);

    tileNum =
        std::max<int64_t>(tileNum, 1);

    // 13. 下发参数
    tiling->totalLength =
        static_cast<uint32_t>(totalLength);

    tiling->blockLength =
        static_cast<uint32_t>(blockLength);

    tiling->tileLength =
        static_cast<uint32_t>(tileLength);

    tiling->tileNum =
        static_cast<uint32_t>(tileNum);

    // 14. 配置启动核数
    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));

    OP_LOGD(
        context,
        "Relu Round5: total=%ld core=%ld block=%ld tile=%ld tileNum=%ld",
        static_cast<long>(totalLength),
        static_cast<long>(usedCoreNum),
        static_cast<long>(blockLength),
        static_cast<long>(tileLength),
        static_cast<long>(tileNum));

    return ge::GRAPH_SUCCESS;
}

// Tiling Parse
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
