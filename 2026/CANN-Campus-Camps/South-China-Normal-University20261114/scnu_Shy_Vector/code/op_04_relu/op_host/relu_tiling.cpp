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
constexpr int64_t SINGLE_CORE_MAX_BYTES = 128 * 1024; // 数据量（字节）低于该阈值时使用单核，避免多核调度开销
constexpr int64_t BLOCK_BYTE_SIZE = 32;      // 256bit，GM 与 UB 之间对齐搬运的块粒度
constexpr int64_t UB_RESERVED_BYTES = 2048;  // UB 预留字节数，覆盖对齐补齐与接口内部临时空间
constexpr int64_t FLOAT_TYPE_SIZE = 4;
constexpr int64_t HALF_TYPE_SIZE = 2;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
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
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const bool isHalfType =
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16);
    const int64_t typeSize = isHalfType ? HALF_TYPE_SIZE : FLOAT_TYPE_SIZE;
    // 每核 GM 起始地址、每次 UB 循环的长度都按 32B 对齐切分，尾块走非对齐通路
    const int64_t alignElements = BLOCK_BYTE_SIZE / typeSize;

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = EnsureNotScalar(inputShape->GetOriginShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "invalid dim %ld", dim), return ge::GRAPH_FAILED);
        totalNum *= dim;
    }

    const int64_t totalBytes = totalNum * typeSize;
    int64_t blockDim = 1;
    int64_t blockFactor = CeilAlign(totalNum, alignElements);
    if (totalBytes > SINGLE_CORE_MAX_BYTES) { // 数据量足够大时才多核切分，避免核间调度开销超过收益
        blockDim = coreNum;
        blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignElements);
        blockDim = CeilDiv(totalNum, blockFactor); // 对齐后每核数据量变大，实际使用的核数可能减少
        if (blockDim < 1 || blockDim > coreNum) {
            blockDim = coreNum;
        }
    }

    // UB 预算：输入/输出队列各若干份 buffer，双缓冲下峰值占用约 (IN+OUT)*typeSize 字节/元素
    const int64_t ubBudget = static_cast<int64_t>(ubSize) * 9 / 10;
    const int64_t totalBuffers = 2 + 2; // 与 kernel 侧 IN_BUFFER_NUM/OUT_BUFFER_NUM 保持一致
    const int64_t bytesPerElement = totalBuffers * typeSize;
    int64_t ubFactor = (ubBudget - UB_RESERVED_BYTES) / bytesPerElement / alignElements * alignElements;
    ubFactor = ubFactor < blockFactor ? ubFactor : blockFactor; // 单次 UB 循环不需要超过本核数据量
    if (ubFactor < alignElements) {
        ubFactor = alignElements;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(blockDim);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isHalfType) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
