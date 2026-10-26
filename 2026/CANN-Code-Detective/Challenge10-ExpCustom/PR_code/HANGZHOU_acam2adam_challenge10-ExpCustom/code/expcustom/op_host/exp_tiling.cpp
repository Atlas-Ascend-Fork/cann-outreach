/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;
// 双缓冲下 UB 中 tensor 数量：输入 2 块 + 输出 2 块
constexpr int64_t BUFFER_TENSOR_COUNT = 4;

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

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    // totalNum 由输入 shape 逐维相乘得到
    totalNum = 1;
    for (size_t i = 0; i < inputShapeX.GetDimNum(); i++) {
        totalNum *= inputShapeX.GetDim(i);
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Exp only supports DT_FLOAT and DT_FLOAT16"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 即使 workspace 大小为 0，也必须声明 workspace slot
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetShapeInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = ExpTilingData();

    // 空 Tensor 场景，避免除零
    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(
            static_cast<uint64_t>(dataType == ge::DT_FLOAT16 ? EXP_TPL_SCH_MODE_0 : EXP_TPL_SCH_MODE_1)));
        return ge::GRAPH_SUCCESS;
    }

    const int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
    const int64_t alignElem = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;

    // 多核切分：按 AIV 核数均分，并按 32B 对齐
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElem);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);
    OP_CHECK_IF(usedCoreNum <= 0, OP_LOGE(context, "usedCoreNum is 0"), return ge::GRAPH_FAILED);

    // UB 切分：双缓冲下输入输出共 4 块 tensor，按 32B 对齐向下取整
    tiling->ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize) / typeSize, BUFFER_TENSOR_COUNT), alignElem);
    if (tiling->ubFactor < alignElem) {
        tiling->ubFactor = alignElem;
    }

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey: float16 -> 0, float32 -> 1
    context->SetTilingKey(GET_TPL_TILING_KEY(
        static_cast<uint64_t>(dataType == ge::DT_FLOAT16 ? EXP_TPL_SCH_MODE_0 : EXP_TPL_SCH_MODE_1)));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
