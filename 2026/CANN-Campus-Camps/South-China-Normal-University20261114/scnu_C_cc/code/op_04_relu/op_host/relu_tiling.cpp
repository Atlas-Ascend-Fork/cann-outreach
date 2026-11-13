
/*!
 * \file relu_tiling.cpp
 * \brief Optimized Relu tiling V3
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;

// 小张量策略
constexpr int64_t SMALL_TENSOR_SIZE = 16384;
constexpr int64_t SMALL_CORE_NUM = 16;

// 大张量使用 Double Buffer
constexpr int64_t DOUBLE_BUFFER_NUM = 2;

// 最大单 Tile 容量：128KB
constexpr int64_t MAX_TILE_BYTES = 128 * 1024;

// 预留 UB 空间
constexpr int64_t UB_RESERVED_BYTES = 32 * 1024;

// GM 32B 对齐
constexpr int64_t ALIGN_BYTES = 32;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv();

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

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    // Workspace
    size_t* workspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE;

    // 获取输入 Shape
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    int64_t totalNum =
        inputShape->GetStorageShape().GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "Invalid tensor size"),
        return ge::GRAPH_FAILED);

    // 获取输入数据类型
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    auto dtype = inputDesc->GetDataType();

    OP_CHECK_IF(
        dtype != ge::DT_FLOAT &&
        dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Unsupported dtype"),
        return ge::GRAPH_FAILED);

    const int64_t typeSize =
        (dtype == ge::DT_FLOAT16) ? 2 : 4;

    const int64_t alignElements =
        ALIGN_BYTES / typeSize;

    // 1. 根据输入规模选择启动核数
    int64_t usedCoreNum;

    if (totalNum <= SMALL_TENSOR_SIZE) {
        usedCoreNum = SMALL_CORE_NUM;
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
    } else {
        usedCoreNum = coreNum;
    }

    int64_t maxUsefulCores =
        CeilDiv(totalNum, alignElements);

    if (usedCoreNum > maxUsefulCores) {
        usedCoreNum = maxUsefulCores;
    }

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // 2. 每核处理元素数
    int64_t blockFactor = CeilAlign(
        CeilDiv(totalNum, usedCoreNum),
        alignElements);

    // 3. 小张量单缓冲，大张量双缓冲
    const int64_t bufferNum =
        (totalNum <= SMALL_TENSOR_SIZE)
            ? 1
            : DOUBLE_BUFFER_NUM;

    // 4. 计算 UB 可用空间
    int64_t usableUbSize =
        static_cast<int64_t>(ubSize);

    if (usableUbSize > UB_RESERVED_BYTES) {
        usableUbSize -= UB_RESERVED_BYTES;
    }

    int64_t maxUbFactor =
        usableUbSize / (bufferNum * typeSize);

    int64_t maxTileFactor =
        MAX_TILE_BYTES / typeSize;

    // 5. 计算 Tile 大小
    int64_t ubFactor = blockFactor;

    if (ubFactor > maxUbFactor) {
        ubFactor = maxUbFactor;
    }

    if (ubFactor > maxTileFactor) {
        ubFactor = maxTileFactor;
    }

    ubFactor = FloorAlign(
        ubFactor,
        alignElements);

    OP_CHECK_IF(
        ubFactor <= 0,
        OP_LOGE(context, "Insufficient UB"),
        return ge::GRAPH_FAILED);

    // 6. 写入 Tiling 数据
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 7. 启动核数
    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));

    // 8. Tiling Key
    uint64_t tilingKey;

    if (dtype == ge::DT_FLOAT16) {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(tilingKey);

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
