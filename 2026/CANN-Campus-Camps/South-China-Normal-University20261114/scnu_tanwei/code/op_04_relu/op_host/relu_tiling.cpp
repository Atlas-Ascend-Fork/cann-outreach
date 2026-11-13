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
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// UB 中同时存在输入/输出两块队列，每块队列开启 double buffer
constexpr int64_t BUFFER_NUM = 2;
// UB 预算占比，预留一部分给框架/入栈数据，避免越界
constexpr int64_t UB_BUDGET_NUM = 3;
constexpr int64_t UB_BUDGET_DEN = 4;

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

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    int64_t typeSize = isFp16 ? 2 : 4;

    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    int64_t totalNum = inputStorageShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "totalNum is negative"), return ge::GRAPH_FAILED);

    int64_t blockDim = 1;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;

    if (totalNum > 0) {
        // 32 Byte 对齐，保证 DataCopy 搬运高效且地址对齐
        int64_t alignElem = 32 / typeSize;
        if (alignElem <= 0) {
            alignElem = 1;
        }

        // 1. 多核切分：优先使用全部 AIV 核，向上对齐后回算实际使用核数
        blockDim = coreNum;
        blockFactor = CeilDiv(totalNum, blockDim);
        blockFactor = CeilAlign(blockFactor, alignElem);

        // 小数据量时避免启动过多空核
        if (blockFactor > 0) {
            blockDim = CeilDiv(totalNum, blockFactor);
        }
        if (blockDim < 1) {
            blockDim = 1;
        }

        // 2. 单核内 UB 切分：输入+输出队列、double buffer 共 4 份缓存
        int64_t ubBudget = static_cast<int64_t>(ubSize) * UB_BUDGET_NUM / UB_BUDGET_DEN;
        int64_t maxUbElem = ubBudget / (typeSize * 2 * BUFFER_NUM);
        maxUbElem = FloorAlign(maxUbElem, alignElem);
        if (maxUbElem < alignElem) {
            maxUbElem = alignElem;
        }

        ubFactor = (blockFactor < maxUbElem) ? blockFactor : maxUbElem;
        ubFactor = FloorAlign(ubFactor, alignElem);
        if (ubFactor < alignElem) {
            ubFactor = alignElem;
        }
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey：MODE_0 -> half，MODE_1 -> float
    uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(isFp16 ? RELU_TPL_SCH_MODE_0 : RELU_TPL_SCH_MODE_1));
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
