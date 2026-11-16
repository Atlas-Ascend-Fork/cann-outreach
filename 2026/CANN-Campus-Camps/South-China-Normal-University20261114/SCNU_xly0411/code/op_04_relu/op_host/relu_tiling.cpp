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

// 与 op_kernel/relu.h 保持一致：输入/输出队列各 BUFFER_NUM(2) 块 DoubleBuffer
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t UB_BLOCK_BYTES = 32;           // UB 数据块大小
constexpr int64_t UB_RESERVED_BYTES = 32 * 1024; // 预留 UB，避免顶满

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

    // 输入 dtype 决定 UB 对齐粒度：float 4B（8 元素/32B），half 2B（16 元素/32B）
    auto inputDesc = context->GetInputDesc(0);
    const bool isFloat32 = (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT);
    const int64_t typeSize = isFloat32 ? 4 : 2;
    const int64_t alignNum = UB_BLOCK_BYTES / typeSize;

    // 元素总数（与输入 shape 相同）
    int64_t totalNum = 0;
    auto inputShape = context->GetInputShape(0);
    if (inputShape != nullptr) {
        totalNum = inputShape->GetOriginShape().GetShapeSize();
    }
    if (totalNum < 0) {
        totalNum = 0;
    }

    int64_t blockFactor = alignNum;
    int64_t ubFactor = alignNum;
    int64_t usedCoreNum = 1;
    if (totalNum > 0) {
        // 每个核处理的元素数，向上按 32B 对齐
        blockFactor = (totalNum + coreNum - 1) / coreNum;
        blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
        usedCoreNum = (totalNum + blockFactor - 1) / blockFactor;
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }

        // 单元素 UB 占用：输入 DoubleBuffer 2 份 + 输出 DoubleBuffer 2 份
        const int64_t perElemBytes = BUFFER_NUM * 2 * typeSize;
        int64_t usableUbSize = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
        int64_t maxUbFactor = usableUbSize / perElemBytes;
        if (maxUbFactor < alignNum) {
            maxUbFactor = alignNum;
        }
        // UB 内分块数（tileNum）由 ubFactor 决定：ubFactor 越大，每核循环次数越少
        ubFactor = maxUbFactor / alignNum * alignNum;
        if (ubFactor > blockFactor) {
            ubFactor = blockFactor;
        }
        if (ubFactor < alignNum) {
            ubFactor = alignNum;
        }
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 配置启动核数量
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto desc = context->GetInputDesc(0);
    if (desc != nullptr && (desc->GetDataType() == ge::DT_FLOAT16 || desc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling