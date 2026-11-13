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
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

    // 获取 dtype，决定 typeSize 与 tilingKey
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : TYPE_SIZE;

    // 总元素数（用 int64 累乘，避免溢出）
    int64_t totalNum = 1;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    for (size_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= x_shape->GetStorageShape().GetDim(i);
    }
    if (totalNum <= 0) {
        totalNum = 1;
    }

    // 32 字节对齐的元素数（UB 搬运的基本粒度）
    int64_t alignNum = 32 / typeSize;
    if (alignNum < 1) {
        alignNum = 1;
    }

    // 多核切分：每核 blockFactor 向上取整到 alignNum 的整数倍
    int64_t blockFactor = (totalNum + coreNum - 1) / coreNum;
    blockFactor = ((blockFactor + alignNum - 1) / alignNum) * alignNum;
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }

    // 实际使用的核数：不能超过需要覆盖 totalNum 的核数，也不能超过 coreNum
    int64_t usedCoreNum = (totalNum + blockFactor - 1) / blockFactor;
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // UB 分块：每核按 ubFactor 循环，ubFactor 也必须是 alignNum 的整数倍
    // 先按每核分 2 块估算（配合 DoubleBuffer），再受 UB 容量约束
    int64_t ubFactor = blockFactor / 2;
    ubFactor = (ubFactor / alignNum) * alignNum;
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    // UB 容量约束：输入 + 输出各 BUFFER_NUM 份
    // BUFFER_NUM = 2，所以总计 2 * 2 * ubFactor * typeSize <= ubSize
    int64_t maxUbFactor = static_cast<int64_t>(ubSize) / (4 * typeSize);
    maxUbFactor = (maxUbFactor / alignNum) * alignNum;
    if (maxUbFactor < alignNum) {
        maxUbFactor = alignNum;
    }
    if (ubFactor > maxUbFactor) {
        ubFactor = maxUbFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // tilingKey：float16/bf16 -> MODE_0，float -> MODE_1
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
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