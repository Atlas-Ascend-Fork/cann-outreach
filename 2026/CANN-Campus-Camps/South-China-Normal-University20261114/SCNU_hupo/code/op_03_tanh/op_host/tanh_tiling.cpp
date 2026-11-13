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
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;

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
    // 阶段1：获取平台信息（UB 大小、AIV 核数）
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 阶段2：获取输入 shape 与 dtype，计算总元素数
    const gert::StorageShape* inputShapeX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeX);
    int64_t totalIdx = 1;
    for (int64_t i = 0; i < inputShapeX->GetStorageShape().GetDimNum(); i++) {
        totalIdx *= inputShapeX->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(totalIdx <= 0, OP_LOGE(context, "totalIdx is invalid"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "invalid dtype"),
        return ge::GRAPH_FAILED);
    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;

    // 阶段3：设置 Workspace 大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 阶段4：计算 tiling 数据并下发到 kernel 侧
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // totalNum：总元素数
    tiling->totalNum = totalIdx;
    // blockFactor：每核处理的元素数（上限 8 核；测试数据量下 8 核每核整除最优）
    constexpr int64_t MAX_CORE_NUM = 8;
    int64_t effCoreNum = (coreNum > MAX_CORE_NUM) ? MAX_CORE_NUM : coreNum;
    tiling->blockFactor = Ops::Base::CeilDiv(totalIdx, effCoreNum);
    // 实际使用的核数（尾核只处理剩余部分，kernel 侧据此做边界处理）
    int64_t usedCoreNum = Ops::Base::CeilDiv(totalIdx, tiling->blockFactor);
    context->SetBlockDim(usedCoreNum);

    // ubFactor：UB 单次循环处理的元素数
    // 输入 2 块 + 输出 2 块（dtype 类型）+ 中间变量 2 块（fp32）=
    // 每元素 4*typeSize + 8 字节；向下对齐到 32 字节（满足矢量指令对齐要求）
    int64_t bytesPerElem = 4 * typeSize + 2 * static_cast<int64_t>(sizeof(float));
    int64_t alignElems = 32 / typeSize;
    tiling->ubFactor = Ops::Base::FloorAlign(
        Ops::Base::FloorDiv(static_cast<int64_t>(ubSize), bytesPerElem), alignElems);
    OP_CHECK_IF(tiling->ubFactor <= 0, OP_LOGE(context, "ubFactor is invalid"), return ge::GRAPH_FAILED);

    // 中间计算变量（fp32 两块）大小已计入 bytesPerElem，此处仅保留字段占位
    tiling->tmpBufSize = 0;

    // 根据输入 dtype 选择 tilingKey（MODE_0 -> half，MODE_1 -> float）
    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
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
