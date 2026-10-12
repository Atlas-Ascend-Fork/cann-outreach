/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;              // 最大元素字节数，按 float32 预算 UB
constexpr int64_t MIN_SPLIT_THRESHOLD = 2048;  // 每个核至少处理的元素数量，控制块数上限，降低块调度开销
constexpr int64_t UB_QUEUE_NUM = 2;           // x / y 两个队列
constexpr int64_t UB_BUFFER_NUM = 2;          // 每个队列的双缓冲深度，与 kernel 侧 BUFFER_NUM 保持一致
constexpr int64_t UB_TMP_RESERVE = 2;         // 预留一半 UB 供高阶 API 内部临时空间使用

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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
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

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 输入为 AutoContiguous 的 ND 张量，总元素数按维度累乘（标量按 1 处理）
    const gert::Shape inShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < inShape.GetDimNum(); i++) {
        totalNum *= inShape.GetDim(i);
    }
    tiling->totalNum = totalNum;

    if (totalNum <= 0) {
        // 空张量：kernel 侧按 0 元素处理
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
    } else {
        // 每核至少 MIN_SPLIT_THRESHOLD 个元素：小输入少用核，大输入至多用满所有核
        int64_t usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
        usedCoreNum = usedCoreNum < 1 ? 1 : usedCoreNum;
        usedCoreNum = usedCoreNum < coreNum ? usedCoreNum : coreNum;
        // 每个核处理的元素数量
        int64_t blockFactor = CeilDiv(totalNum, usedCoreNum);
        tiling->blockFactor = blockFactor;
        // x/y 两个队列各 BUFFER_NUM 份双缓冲，且预留一半 UB 给高阶 API 内部临时空间
        int64_t maxUbElem = static_cast<int64_t>(ubSize) /
                            (UB_TMP_RESERVE * UB_QUEUE_NUM * UB_BUFFER_NUM * TYPE_SIZE);
        // 每次 UB 循环处理的元素数量
        tiling->ubFactor = blockFactor < maxUbElem ? blockFactor : maxUbElem;
        context->SetBlockDim(static_cast<uint32_t>(CeilDiv(totalNum, blockFactor)));
    }

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
