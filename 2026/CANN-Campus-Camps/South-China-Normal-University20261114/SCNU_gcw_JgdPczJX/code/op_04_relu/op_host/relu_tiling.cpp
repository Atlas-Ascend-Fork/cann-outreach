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

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;

constexpr uint32_t WS_SYS_SIZE = 16 * 1024 * 1024;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t SMALL_DATA_THRESHOLD = 65536;

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
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Platform error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Workspace error"), return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t typeSize = ge::GetSizeByDataType(inputDesc->GetDataType());

    tiling->totalNum = totalNum;
    
    int64_t align = 32 / typeSize;
    int64_t usedCoreNum = coreNum;

    if (totalNum <= MIN_SPLIT_THRESHOLD) {
        usedCoreNum = 1;
        tiling->blockFactor = totalNum;
        tiling->ubFactor = totalNum;
    } else if (totalNum <= SMALL_DATA_THRESHOLD) {
        usedCoreNum = 8;
        int64_t blockFactor = (totalNum + usedCoreNum - 1) / usedCoreNum;
        blockFactor = (blockFactor + align - 1) / align * align;
        tiling->blockFactor = blockFactor;
        tiling->ubFactor = blockFactor; 
    } else {
        int64_t blockFactor = (totalNum + coreNum - 1) / coreNum;
        blockFactor = (blockFactor + align - 1) / align * align;
        tiling->blockFactor = blockFactor;

        usedCoreNum = (totalNum + blockFactor - 1) / blockFactor;
        if (usedCoreNum > coreNum) usedCoreNum = coreNum;

        int64_t maxUbElements = 8192;
        if (blockFactor < maxUbElements) {
            tiling->ubFactor = blockFactor;
        } else {
            int64_t ubFactor = blockFactor / 4; 
            ubFactor = (ubFactor + align - 1) / align * align;
            if (ubFactor > maxUbElements) {
                ubFactor = maxUbElements;
            }
            tiling->ubFactor = ubFactor;
        }
    }

    context->SetBlockDim(usedCoreNum);

    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context) 
{ 
    return ge::GRAPH_SUCCESS; 
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling