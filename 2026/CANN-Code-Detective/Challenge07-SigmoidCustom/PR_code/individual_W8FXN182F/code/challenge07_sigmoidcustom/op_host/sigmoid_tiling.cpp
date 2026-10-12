/*!
 * \file sigmoid_tiling.cpp
 * \brief Sigmoid 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sigmoid_tiling_data.h"
#include "../op_kernel/sigmoid_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 数据搬运 32B 对齐：按最小 dtype（half, 2B）折算 16 元素，对 fp32（8 元素）同样满足
constexpr int64_t ALIGN_ELEMS = 16;

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

static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
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

    SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 总元素数（int64 累乘防溢出）
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= xShape->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "invalid totalNum: %ld", (long)totalNum),
        return ge::GRAPH_FAILED);

    // 多核切分：优先用满核数，但每核不少于 MIN_SPLIT_THRESHOLD 个元素（摊薄调度开销）
    int64_t usedCoreNum = coreNum;
    if (totalNum < static_cast<int64_t>(MIN_SPLIT_THRESHOLD) * coreNum) {
        usedCoreNum = CeilDiv(totalNum, static_cast<int64_t>(MIN_SPLIT_THRESHOLD));
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    // 每核标准处理量向上取整并对齐 16 元素，保证满块 DataCopy 32B 对齐
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), ALIGN_ELEMS);
    usedCoreNum = CeilDiv(totalNum, blockFactor);
    // 尾核实际处理量（可能小于 blockFactor）
    int64_t lastBlockLength = totalNum - blockFactor * (usedCoreNum - 1);
    if (lastBlockLength <= 0) {
        lastBlockLength = blockFactor;
    }

    // 单次 UB 预算按输入 dtype 区分：
    // half: in+out 双缓冲 8B/元素 + fp32 tmp/one 8B/元素 = 16B/元素
    // float: in+out 双缓冲 16B/元素 + fp32 tmp/one 8B/元素 = 24B/元素
    auto inputDescForUb = context->GetInputDesc(0);
    const int64_t bytesPerElem =
        (inputDescForUb != nullptr && inputDescForUb->GetDataType() == ge::DT_FLOAT) ? 24 : 16;
    int64_t maxUbElems = FloorAlign(
        static_cast<int64_t>(ubSize) / bytesPerElem, ALIGN_ELEMS);
    int64_t ubFactor = blockFactor;
    if (ubFactor > maxUbElems) {
        ubFactor = maxUbElems;
    }
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->lastBlockLength = lastBlockLength;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SigmoidCompileInfo {};

IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);

} // namespace optiling
