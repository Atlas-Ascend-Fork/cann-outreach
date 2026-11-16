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
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BUFFER_NUM = 2; // 双缓冲

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

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    int64_t totalNum = 1;
    const gert::Shape& shape = xShape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalNum *= shape.GetDim(i);
    }

    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }
    tiling->totalNum = totalNum;

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    bool isFp16 = (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16);
    int64_t typeSize = isFp16 ? 2 : 4;
    
    // 32B 对齐用于计算与搬运安全，4KB 对齐用于 DMA 极速 burst 搬运
    int64_t alignNum = 32 / typeSize;
    int64_t dmaAlignNum = 4096 / typeSize; 

    // 1. 精确计算 ubFactor：扣除 2KB 系统保留余量，其余全部分配给双缓冲
    int64_t ubSizeAvail = static_cast<int64_t>(ubSize) - 2048; 
    int64_t ubFactor = ubSizeAvail / (2 * BUFFER_NUM * typeSize);
    ubFactor = FloorAlign(ubFactor, dmaAlignNum); // 对齐到 4KB 边界，提升搬运效率
    if (ubFactor < alignNum) ubFactor = alignNum;
    tiling->ubFactor = ubFactor;

    // 2. 动态确定核数：小 Shape 尽量少开核，避免 Sync 开销
    int64_t usedCoreNum = CeilDiv(totalNum, ubFactor);
    if (usedCoreNum > coreNum) usedCoreNum = coreNum;
    if (usedCoreNum < 1) usedCoreNum = 1;

    // 3. 计算每核处理量，同样对齐到 4KB 边界，确保主循环每次搬运都整齐
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), dmaAlignNum);
    usedCoreNum = CeilDiv(totalNum, blockFactor); // 反推真实使用的核数

    tiling->blockFactor = blockFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    uint64_t tilingKey = isFp16 ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
                                : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
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