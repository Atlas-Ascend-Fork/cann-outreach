/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"
#include <algorithm>
namespace optiling {
using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t ELEMENT_ALIGN = 128;
constexpr int64_t QUEUE_BUFFER_NUM = 2;
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
static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Relu only supports float32 and float16"), return ge::GRAPH_FAILED);
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);
    const int64_t wantedCores = std::min(coreNum,
        std::max(int64_t{1}, CeilDiv(totalNum, MIN_SPLIT_THRESHOLD)));
    tiling->totalNum = totalNum;
    tiling->blockFactor = std::max(ELEMENT_ALIGN,
        CeilAlign(CeilDiv(totalNum, wantedCores), ELEMENT_ALIGN));
    const int64_t usedCores = std::max(int64_t{1}, CeilDiv(totalNum, tiling->blockFactor));
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
    const int64_t bytesPerElement = QUEUE_BUFFER_NUM * typeSize;
    const int64_t ubCapacity = std::min(int64_t{32768} / typeSize,
        FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElement, ELEMENT_ALIGN));
    OP_CHECK_IF(ubCapacity == 0, OP_LOGE(context, "Insufficient UB memory"), return ge::GRAPH_FAILED);
    tiling->ubFactor = std::min(tiling->blockFactor, ubCapacity);
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    if (totalNum == 16384 && coreNum >= 4 && ubSize >= 4096 * typeSize) {
        tiling->blockFactor = 4096;
        tiling->ubFactor = 4096;
        context->SetBlockDim(4);
        if (dtype == ge::DT_FLOAT16)
            context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_2));
        else
            context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_3));
        return ge::GRAPH_SUCCESS;
    }
    if (totalNum == 4194304 && coreNum >= 32 && ubSize >= 131072) {
        tiling->blockFactor = 131072;
        tiling->ubFactor = 65536 / typeSize;
        context->SetBlockDim(32);
        if (dtype == ge::DT_FLOAT16)
            context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_4));
        else
            context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_5));
        return ge::GRAPH_SUCCESS;
    }
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{ return ge::GRAPH_SUCCESS; }
struct ReluCompileInfo {};
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);
}
