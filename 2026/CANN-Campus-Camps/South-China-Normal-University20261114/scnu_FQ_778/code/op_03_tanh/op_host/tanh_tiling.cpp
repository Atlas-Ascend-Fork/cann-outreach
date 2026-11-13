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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024; // 每个核至少处理的元素个数
constexpr int64_t BUFFER_NUM = 2;             // 与 kernel 侧双缓冲保持一致
constexpr int64_t BLOCK_BYTES = 32;           // UB 地址/长度对齐字节数
constexpr int64_t CORE_NUM_MIN = 1;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static inline int64_t CeilDivInt(int64_t num1, int64_t num2)
{
    if (num2 <= 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

static inline int64_t CeilAlignInt(int64_t num1, int64_t num2)
{
    if (num2 <= 0) {
        return num1;
    }
    return CeilDivInt(num1, num2) * num2;
}

static inline int64_t FloorAlignInt(int64_t num1, int64_t num2)
{
    if (num2 <= 0) {
        return num1;
    }
    return num1 / num2 * num2;
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

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const bool isSmallType = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    const int64_t typeSize = isSmallType ? 2 : TYPE_SIZE;
    const int64_t alignNum = (BLOCK_BYTES / typeSize) > 0 ? (BLOCK_BYTES / typeSize) : 1;

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& inShape = EnsureNotScalar(xShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < inShape.GetDimNum(); ++i) {
        const int64_t dim = inShape.GetDim(i);
        OP_CHECK_IF(
            dim < 0, OP_LOGE(context, "tanh does not support dynamic shape, dim is %ld", dim),
            return ge::GRAPH_FAILED);
        totalNum *= dim;
    }
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    int64_t usedCoreNum = (coreNum > CORE_NUM_MIN) ? coreNum : CORE_NUM_MIN;
    const int64_t coreNumByThreshold = CeilDivInt((totalNum > 1) ? totalNum : 1, MIN_SPLIT_THRESHOLD);
    if (usedCoreNum > coreNumByThreshold) {
        usedCoreNum = coreNumByThreshold;
    }
    if (usedCoreNum < CORE_NUM_MIN) {
        usedCoreNum = CORE_NUM_MIN;
    }

    int64_t blockFactor = CeilAlignInt(CeilDivInt(totalNum, usedCoreNum), alignNum);
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }

    usedCoreNum = CeilDivInt(totalNum, blockFactor);
    if (usedCoreNum < CORE_NUM_MIN) {
        usedCoreNum = CORE_NUM_MIN;
    }

    const int64_t ubBudget = static_cast<int64_t>(ubSize) / 2;
    int64_t maxUbNum = ubBudget / (typeSize * BUFFER_NUM * 2);
    maxUbNum = FloorAlignInt(maxUbNum, alignNum);
    if (maxUbNum < alignNum) {
        maxUbNum = alignNum;
    }
    int64_t ubFactor = (blockFactor < maxUbNum) ? blockFactor : maxUbNum;
    ubFactor = FloorAlignInt(ubFactor, alignNum);
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    uint64_t tilingKey = isSmallType ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
                                     : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    OP_LOGI(context, "Tanh tiling: totalNum = %lld, blockFactor = %lld, ubFactor = %lld, blockDim = %lld, "
                     "tilingKey = %llu",
            static_cast<long long>(totalNum), static_cast<long long>(blockFactor),
            static_cast<long long>(ubFactor), static_cast<long long>(usedCoreNum),
            static_cast<unsigned long long>(tilingKey));
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


