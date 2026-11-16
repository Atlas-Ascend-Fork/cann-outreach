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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ------------------------------------------------------------------
    // 1. 取输入信息：元素总数与数据类型
    // ------------------------------------------------------------------
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    int64_t totalNum = inputShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "totalNum is negative"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    // float32 占 4 字节，float16 / bfloat16 占 2 字节
    int64_t typeSize = TYPE_SIZE;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        typeSize = TYPE_SIZE / 2;
    }

    // ------------------------------------------------------------------
    // 2. 根据 dtype 决定 tilingKey：fp16 走 schMode 0，fp32 走 schMode 1
    // ------------------------------------------------------------------
    bool isHalf = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    uint64_t tilingKey = isHalf ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
                                : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);

    // 空张量直接返回，避免后续除以 0
    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 1;
        context->SetBlockDim(1);
        context->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    // ------------------------------------------------------------------
    // 3. 多核切分：每核长度向上取整，再按 32B 对齐元素数上取整
    //    核数由硬件决定，总长度未必除得尽，余量由最后一核吸收
    // ------------------------------------------------------------------
    constexpr int64_t UB_BLOCK_SIZE = 32;      // DataCopy / UB 的最小对齐粒度
    constexpr int64_t FLOAT_TMP_BUF_NUM = 5;   // 计算域 float 缓存个数
    constexpr int64_t RESERVED_UB_SIZE = 32 * 1024;  // 预留给框架的 UB 余量

    int64_t alignNum = UB_BLOCK_SIZE / typeSize;  // 32B 对应的元素个数：fp32=8，fp16=16

    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    blockFactor = CeilAlign(blockFactor, alignNum);

    // 实际需要拉起的核数：数据量小时不必把全部核叫醒
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // ------------------------------------------------------------------
    // 4. UB 预算：输入/输出双缓冲各 2 块，外加 5 块 float 计算缓存
    // ------------------------------------------------------------------
    int64_t perElemBytes = 4 * typeSize + FLOAT_TMP_BUF_NUM * TYPE_SIZE;
    int64_t ubBudget = static_cast<int64_t>(ubSize) - RESERVED_UB_SIZE;
    int64_t ubFactor = (ubBudget > 0) ? (ubBudget / perElemBytes) : blockFactor;
    ubFactor = FloorAlign(ubFactor, alignNum);
    if (ubFactor > blockFactor) {
        // 单核总量都不够装一趟，无需再切细
        ubFactor = blockFactor;
    }
    if (ubFactor <= 0) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
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
