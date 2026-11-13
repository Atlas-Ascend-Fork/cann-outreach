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
constexpr int64_t QUEUE_NUM = 2;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t TEMP_BUFFER_NUM = 4;   // kernel 侧 4 个 fp32 临时 buffer
constexpr int64_t MAX_UB_FACTOR = 2048;  // 与 kernel 侧单次 UB 计算上限一致

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

    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const int64_t totalNum = EnsureNotScalar(inputStorageShape->GetOriginShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input element count"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    const int64_t typeSize = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) ? 2 : TYPE_SIZE;
    const int64_t alignNum = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;
    OP_CHECK_IF(alignNum <= 0, OP_LOGE(context, "Invalid UB alignment"), return ge::GRAPH_FAILED);

    // 小输入减少核调度开销；核内起始地址按 32 字节对齐。
    int64_t usedCoreNum = 1;
    int64_t blockFactor = alignNum;
    if (totalNum > 0) {
        const int64_t requiredCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
        usedCoreNum = coreNum < requiredCoreNum ? coreNum : requiredCoreNum;
        blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
        usedCoreNum = CeilDiv(totalNum, blockFactor);
    }

    // 输入输出队列各 DoubleBuffer，另加 4 个 fp32 临时 buffer。
    const int64_t bytesPerElem = typeSize * QUEUE_NUM * BUFFER_NUM + TYPE_SIZE * TEMP_BUFFER_NUM;
    const int64_t maxUbFactor = FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElem, alignNum);
    OP_CHECK_IF(maxUbFactor < alignNum, OP_LOGE(context, "UB size is too small"), return ge::GRAPH_FAILED);
    int64_t ubFactor = blockFactor < maxUbFactor ? blockFactor : maxUbFactor;
    if (ubFactor > MAX_UB_FACTOR) {
        ubFactor = FloorAlign(MAX_UB_FACTOR, alignNum);
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey。
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
