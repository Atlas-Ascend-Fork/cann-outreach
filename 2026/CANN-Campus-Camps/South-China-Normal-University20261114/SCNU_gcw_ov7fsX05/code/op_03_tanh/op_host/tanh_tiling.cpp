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
constexpr int64_t TYPE_SIZE_FP32 = 4;
constexpr int64_t TYPE_SIZE_FP16 = 2;
// UB 切分份数：输入 2 块 + 输出 2 块 + 给 Tanh 内部临时空间留 12 份余量
constexpr int64_t UB_SPLIT_NUM = 16;

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
    // ===== 阶段1：平台信息（AIV 核数、UB 大小）=====
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

    // ===== 阶段2：按 dtype 选 tilingKey（schMode: 0 -> half, 1 -> float）=====
    // 注意：本算子的模板参数是 UINT 的 schMode（见 tanh_tiling_key.h），
    //       不能用 ASCENDC_TPL_SEL_PARAM 传 dtype，必须用 GET_TPL_TILING_KEY。
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "only float32/float16 are supported"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0UL;
    int64_t typeSize = TYPE_SIZE_FP32;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
        typeSize = TYPE_SIZE_FP16;
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
        typeSize = TYPE_SIZE_FP32;
    }
    context->SetTilingKey(tilingKey);

    // ===== 阶段3：总元素数（shape 逐维累乘）=====
    const gert::StorageShape* inputShapeX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeX);
    int64_t totalNum = 1;
    const gert::Shape& shapeX = inputShapeX->GetStorageShape();
    for (size_t i = 0; i < shapeX.GetDimNum(); i++) {
        totalNum *= shapeX.GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    // ===== 阶段4：填充 tiling 数据 =====
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 32B 对齐元素数：float -> 8，float16 -> 16
    const int64_t alignElems = 32 / typeSize;

    // 每核处理量：均分后向上对齐到 32B，保证每核 GM 起始地址对齐
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElems);
    OP_CHECK_IF(blockFactor <= 0, OP_LOGE(context, "blockFactor is invalid"), return ge::GRAPH_FAILED);

    // 实际启动核数（尾核只处理剩余部分）
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    OP_CHECK_IF(usedCoreNum <= 0, OP_LOGE(context, "usedCoreNum is invalid"), return ge::GRAPH_FAILED);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 单次 UB 循环处理量：按 UB 实大小反算，向下对齐 32B，且不超过 blockFactor
    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, UB_SPLIT_NUM), alignElems);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor < alignElems) {
        ubFactor = alignElems;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // tiling 数据长度必须显式声明，否则 kernel 侧读到的结构体可能为空
    context->GetRawTilingData()->SetDataSize(sizeof(TanhTilingData));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
