/*!
 * \file leakyrelu_tiling.cpp
 * \brief Leakyrelu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "graph/types.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t UB_BUFFER_NUM = 2;   // input/output 队列深度

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

static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* negativeSlopePtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, negativeSlopePtr);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const int64_t dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(inputDesc->GetDataType()));
    OP_CHECK_IF(
        dtypeSize <= 0,
        OP_LOGE(context, "unsupported dtype size %ld", dtypeSize),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& shape = xShape->GetShape();

    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "input dim %zu is %ld", i, dim), return ge::GRAPH_FAILED);
        totalNum *= dim;
    }

    LeakyreluTilingData* tiling = context->GetTilingData<LeakyreluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    // 清零 padding，保证多次调用时 tiling 数据内容确定
    for (size_t i = 0; i < sizeof(LeakyreluTilingData); ++i) {
        reinterpret_cast<uint8_t*>(tiling)[i] = 0U;
    }

    const int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    const int64_t alignNum = ubBlockSize > 0 ? ubBlockSize / dtypeSize : 1;

    // 输入、输出各一个队列，队列深度 UB_BUFFER_NUM；仅使用一半 UB 空间，
    // 为 AscendC 内部预留空间留出余量
    int64_t ubFactor = (static_cast<int64_t>(ubSize) / 2) / (2 * UB_BUFFER_NUM * dtypeSize);
    ubFactor = FloorAlign(ubFactor, alignNum);
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    int64_t blockFactor = 0;
    uint32_t blockDim = 1U;
    if (totalNum > 0) {
        blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
        blockDim = static_cast<uint32_t>(CeilDiv(totalNum, blockFactor));
        if (blockDim == 0U) {
            blockDim = 1U;
        }
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->negativeSlope = *negativeSlopePtr;

    context->SetBlockDim(blockDim);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLeakyrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LeakyreluCompileInfo {};

IMPL_OP_OPTILING(Leakyrelu).Tiling(LeakyreluTilingFunc).TilingParse<LeakyreluCompileInfo>(TilingParseForLeakyrelu);

} // namespace optiling
