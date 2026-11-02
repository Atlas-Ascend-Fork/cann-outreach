
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

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

// 每个核希望处理的元素数量
constexpr int64_t TARGET_BLOCK_SIZE = 1024;

// 每次搬运到 UB 的元素数量
constexpr int64_t TILE_SIZE = 256;

// 双缓冲
constexpr int64_t BUFFER_NUM = 2;

// 输入输出各需要两份缓冲区
constexpr int64_t QUEUE_NUM = 2;

// float32 最大占 4 字节
constexpr int64_t TYPE_SIZE = 4;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

// 获取硬件平台信息
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(
        context, platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(context, "coreNum is 0"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 配置 workspace
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

// 核心 Tiling 逻辑
static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    // 1. 获取硬件信息
    OP_CHECK_IF(
        GetPlatformInfo(
            context, ubSize, coreNum)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 设置 workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 3. 获取输入 shape
    const auto* inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context, inputShape);

    // 计算输入元素总数
    int64_t totalNum =
        inputShape->GetStorageShape()
                  .GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "Invalid input shape"),
        return ge::GRAPH_FAILED);

    // 4. 获取 TilingData
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context, tiling);

    // 5. 检查 UB 内存是否足够
    uint64_t requiredUbSize =
        static_cast<uint64_t>(
            TILE_SIZE * TYPE_SIZE *
            BUFFER_NUM * QUEUE_NUM);

    OP_CHECK_IF(
        ubSize < requiredUbSize,
        OP_LOGE(context, "UB size is too small"),
        return ge::GRAPH_FAILED);

    // 6. 根据总元素数确定使用的核数
    int64_t blockDim =
        CeilDiv(totalNum, TARGET_BLOCK_SIZE);

    // 不超过硬件可用核数
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }

    if (blockDim < 1) {
        blockDim = 1;
    }

    // 7. 计算每核元素数
    // 按 TILE_SIZE 对齐
    int64_t blockFactor =
        CeilAlign(
            CeilDiv(totalNum, blockDim),
            TILE_SIZE);

    // 重新计算实际需要的核数
    blockDim = CeilDiv(totalNum, blockFactor);

    // 8. 写入 TilingData
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = TILE_SIZE;

    // 9. 设置 AI Core 数量
    context->SetBlockDim(
        static_cast<uint32_t>(blockDim));

    // 10. 根据 dtype 选择 tilingKey
    uint64_t tilingKey;

    auto inputDesc = context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context, inputDesc);

    if (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
        inputDesc->GetDataType() == ge::DT_BF16) {

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0);

    } else {

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(tilingKey);

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
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);

} // namespace optiling
