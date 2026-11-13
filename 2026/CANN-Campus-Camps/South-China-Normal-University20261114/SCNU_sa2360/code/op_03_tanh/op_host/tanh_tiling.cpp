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
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TEMP_BUFFER_NUM = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = inputShape->GetStorageShape();
    int64_t totalNum = 1;
    for (int64_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }

    // 当前核函数使用单核切分 UB；blockFactor 是单核负责的元素数。
    const uint32_t alignBytes = GetUbBlockSize(context);
    const uint32_t alignElems = alignBytes / TYPE_SIZE;
    int64_t ubFactor = static_cast<int64_t>(ubSize / (TYPE_SIZE * (BUFFER_NUM * 2 + TEMP_BUFFER_NUM)));
    ubFactor = FloorAlign(ubFactor, static_cast<int64_t>(alignElems));
    if (ubFactor <= 0) ubFactor = alignElems;
    if (ubFactor > totalNum) ubFactor = FloorAlign(totalNum, static_cast<int64_t>(alignElems));
    if (ubFactor <= 0) ubFactor = totalNum;

    tiling->totalNum = static_cast<uint32_t>(totalNum);
    tiling->blockFactor = static_cast<uint32_t>(totalNum);
    tiling->ubFactor = static_cast<uint32_t>(ubFactor);
    context->SetBlockDim(1);

    auto inputDesc = context->GetInputDesc(0);
    uint64_t tilingKey = (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16)
        ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
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
