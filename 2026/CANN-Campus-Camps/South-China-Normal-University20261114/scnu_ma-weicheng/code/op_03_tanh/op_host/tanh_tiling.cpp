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
constexpr int64_t BUFFER_NUM = 2;      // 与 kernel 侧 TQue 的 buffer 深度保持一致
constexpr int64_t UB_ALIGN_SIZE = 32;  // UB 的 block 大小（字节）

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
    // TODO: 实现 Tiling 逻辑
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

    // 输入信息：元素总数与 dtype（决定对齐粒度与 tilingKey）
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = EnsureNotScalar(xShapePtr->GetStorageShape());
    int64_t totalNum = xShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    const bool isFp16Type = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    int64_t typeSize = isFp16Type ? static_cast<int64_t>(2) : static_cast<int64_t>(TYPE_SIZE);

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    if (totalNum <= 0) {
        // 空 tensor：仍然给出合法 tiling，kernel 侧会因 blockLength_ 为 0 直接返回
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 1;
        context->SetBlockDim(1);
        if (isFp16Type) {
            context->SetTilingKey(GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0));
        } else {
            context->SetTilingKey(GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1));
        }
        return ge::GRAPH_SUCCESS;
    }

    // 32B 对齐下每个 block（32B）能容纳的元素个数
    int64_t alignNum = UB_ALIGN_SIZE / typeSize;
    if (alignNum <= 0) {
        alignNum = 1;
    }

    // 多核切分：每核数据量按 32B 对齐，尾核承担剩余数据
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }

    // UB 单次循环的数据量：
    //   输入、输出各 BUFFER_NUM 份（双缓冲） + 2 份 fp32 中间变量
    int64_t bytesPerElem = static_cast<int64_t>(BUFFER_NUM) * 2 * typeSize + 2 * static_cast<int64_t>(sizeof(float));
    int64_t ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElem, alignNum);
    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }
    if (ubFactor > blockFactor) {
        // 单核数据量不足一个 ubFactor 时无需申请更大的 UB 空间
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey = 0UL;
    if (isFp16Type) {
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
