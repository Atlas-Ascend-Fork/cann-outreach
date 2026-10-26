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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // ① 平台信息：UB 大小 + AIV 核数
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

    // ② 本次要处理的元素总数（运行期从 shape 问出来）
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const int64_t totalNum = xShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    // ③ dtype -> 单个元素字节数（★ 必须放在 ④ 之前，否则用不上）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const bool isFp16 =
        (inputDesc->GetDataType() == ge::DT_FLOAT16) || (inputDesc->GetDataType() == ge::DT_BF16);
    const int64_t typeSize = isFp16 ? 2 : 4;

    // ④ 分核：blockFactor 必须是 32 字节的整数倍，否则每个核的起始地址不对齐，
    //    DataCopy 在真机上会搬错（本地单核 UT 测不出来）
    const int64_t blockAlign = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;
    int64_t blockDim = coreNum;
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    if (totalNum < MIN_SPLIT_THRESHOLD) {
        blockDim = 1;
        blockFactor = totalNum;
    } else {
        blockFactor = CeilAlign(blockFactor, blockAlign);
        blockDim = CeilDiv(totalNum, blockFactor);
    }

    // ⑤ 每趟搬多少：kernel 用 6 份 UB 缓冲（进×2 + 出×2 + 临时×2），故除以 3
    int64_t ubFactor = static_cast<int64_t>(ubSize) / (3 * typeSize);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    const int64_t alignNum = 2 * static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;
    ubFactor = FloorAlign(ubFactor, alignNum);
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor is 0, input is too small"), return ge::GRAPH_FAILED);

    // ⑥ 写 tiling data（host -> device 的唯一通道）
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // ⑦ 起几个核
    context->SetBlockDim(blockDim);

    // ⑧ 走哪条 kernel 实现（fp16 -> MODE_0 -> Tanh<half>；fp32 -> MODE_1 -> Tanh<float>）
    uint64_t tilingKey;
    if (isFp16) {
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