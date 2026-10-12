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
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr size_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;              // 按最大 dtype(float32) 计算 UB 容量
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024; // 每核至少尽量处理 1024 个元素
constexpr int64_t UB_BUFFER_COUNT = 4;        // 1 输入 + 1 输出，均使用双缓冲

static const gert::Shape g_vec_1_shape = {1};

static inline gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    // 0 维标量按 1 元素处理，避免 GetShapeSize 为 0/错误。
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 1. 获取硬件信息：可用 AIV 核数与 UB 大小
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取输入 shape，并计算总元素数
    const gert::StorageShape* inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    gert::Shape inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    int64_t totalNum = inputShapeX.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "invalid totalNum: %ld", totalNum), return ge::GRAPH_FAILED);

    // 3. workspace：ReLU 不需要额外 workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. 计算 tiling 数据
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;

    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    if (blockFactor < MIN_SPLIT_THRESHOLD) {
        blockFactor = (totalNum < MIN_SPLIT_THRESHOLD) ? totalNum : MIN_SPLIT_THRESHOLD;
    }
    tiling->blockFactor = blockFactor;
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // UB 中同时容纳 inputQueueX 双缓冲 + outputQueueY 双缓冲，共 4 份 tensor。
    // TYPE_SIZE 使用 4 字节按 float32 最坏情况计算，因此 float16 场景必然安全。
    int64_t ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "invalid ubBlockSize: %ld", ubBlockSize), return ge::GRAPH_FAILED);

    int64_t maxUbFactor = FloorDiv(static_cast<int64_t>(ubSize) / TYPE_SIZE, UB_BUFFER_COUNT);
    maxUbFactor = FloorAlign(maxUbFactor, ubBlockSize);
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "invalid ubFactor: %ld", maxUbFactor), return ge::GRAPH_FAILED);
    tiling->ubFactor = maxUbFactor;

    context->SetBlockDim(usedCoreNum);

    // 5. 根据输入 dtype 选择 kernel 模板实例（float / float16）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
