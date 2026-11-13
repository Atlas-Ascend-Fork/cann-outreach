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
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t UB_BLOCK_SIZE = 32;      // UB 最小对齐粒度(Byte)
constexpr int64_t UB_RESERVE_FACTOR = 4;   // 双缓冲 2 份 + 计算中间结果预留

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

/* 本地整数工具：向上/向下取整对齐，避免依赖外部工具函数的参数类型 */
static inline int64_t CeilDivInt(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline int64_t CeilAlignInt(int64_t a, int64_t b) { return CeilDivInt(a, b) * b; }
static inline int64_t FloorAlignInt(int64_t a, int64_t b) { return (a / b) * b; }

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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 输入 shape（标量按 {1} 处理），逐维相乘得到总元素数
    const auto* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape& xShape = EnsureNotScalar(xStorageShape->GetStorageShape());
    int64_t totalNum = xShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    // 依据 dtype 确定元素字节数，以及 32B 对齐对应的元素个数
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT) ? TYPE_SIZE : 2;
    int64_t alignNum = UB_BLOCK_SIZE / typeSize;
    OP_CHECK_IF(alignNum <= 0, OP_LOGE(context, "alignNum is invalid"), return ge::GRAPH_FAILED);

    // 多核切分：totalNum 均分到各核，blockFactor 按 32B 对齐
    int64_t blockFactor = CeilAlignInt(CeilDivInt(totalNum, coreNum), alignNum);
    int64_t usedCoreNum = CeilDivInt(totalNum, blockFactor);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    OP_CHECK_IF(usedCoreNum <= 0, OP_LOGE(context, "usedCoreNum is invalid"), return ge::GRAPH_FAILED);

    // UB 切分：为双缓冲和计算中间结果预留空间，同样按 32B 对齐
    int64_t ubFactor = FloorAlignInt(static_cast<int64_t>(ubSize) / (typeSize * UB_RESERVE_FACTOR), alignNum);
    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
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