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

constexpr int64_t TILE_NUM = 1;   // 每核分块数
constexpr int64_t BLOCK_DIM = 8;  // 启动核数
constexpr int64_t kUbSafetyRatio = 2; // UB 预留系数：本算子单块需要 input + output 两份缓冲

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

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t platformCoreNum = 0;
    auto platRet = GetPlatformInfo(context, ubSize, platformCoreNum);
    if (platRet != ge::GRAPH_SUCCESS) {
        return platRet;
    }

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    // 输入 shape 逐维相乘得到总元素数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t dataSz = 1;
    for (int32_t i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        dataSz *= xShape->GetStorageShape().GetDim(i);
    }
    // 按输入 dtype 设置 tilingKey（框架编码：0=float, 1=float16，与 ASCENDC_TPL_DATATYPE_DECL 顺序一致）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    uint32_t dtype = static_cast<uint32_t>(inputDesc->GetDataType());
    context->SetTilingKey((dtype == static_cast<uint32_t>(ge::DT_FLOAT16)) ? 1 : 0);
    int64_t typeSize = static_cast<int64_t>(ge::GetSizeByDataType(static_cast<ge::DataType>(dtype)));
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "invalid input dtype"), return ge::GRAPH_FAILED);

    tiling->totalNum = dataSz;
    // 每核元素数：向上取整，保证非整除 shape 时尾部元素仍有核处理（无静默丢弃）
    tiling->blockFactor = CeilDiv(dataSz, BLOCK_DIM);
    if (tiling->blockFactor < 1) {
        tiling->blockFactor = 1; // 防御：data_sz < 核数
    }
    // 单块元素数：先按每核分块数切分
    tiling->ubFactor = CeilDiv(tiling->blockFactor, TILE_NUM);
    if (tiling->ubFactor < 1) {
        tiling->ubFactor = tiling->blockFactor; // 防御：单块至少 1 元素
    }
    // UB 容量校验：单块需要 input + output 两份缓冲，按实际 dtype 字节宽折算，
    // 超出可用 UB 时收缩到容量允许的最大块（保底 1 个元素）
    int64_t bytesPerElem = kUbSafetyRatio * typeSize;
    int64_t ubCapacityElems = static_cast<int64_t>(ubSize / bytesPerElem);
    if (ubCapacityElems < 1) {
        ubCapacityElems = 1;
    }
    if (tiling->ubFactor > ubCapacityElems) {
        tiling->ubFactor = ubCapacityElems;
    }

    // 配置启动核数
    context->SetBlockDim(BLOCK_DIM);
    // 配置 workspace 大小（本算子无需 workspace）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
