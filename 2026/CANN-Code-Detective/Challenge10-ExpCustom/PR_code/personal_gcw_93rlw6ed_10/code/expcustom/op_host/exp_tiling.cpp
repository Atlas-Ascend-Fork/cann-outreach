/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
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

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- 计算输入总元素数（int64 累乘，防溢出）----
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& inShape = EnsureNotScalar(xShape->GetStorageShape());

    int64_t totalNum = 1;
    const size_t dimNum = inShape.GetDimNum();
    for (size_t i = 0; i < dimNum; ++i) {
        totalNum *= inShape.GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    // ---- dtype 判定：half/bf16 与 float 的 UB 预算不同 ----
    uint64_t tilingKey;
    int64_t inputBytesPerElem = TYPE_SIZE;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
        inputBytesPerElem = sizeof(uint16_t);
    } else {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
        inputBytesPerElem = TYPE_SIZE;
    }

    // ---- 多核切分：小张量缩核，避免空核 ----
    int64_t usedCoreNum = coreNum;
    if (totalNum < MIN_SPLIT_THRESHOLD) {
        usedCoreNum = 1;
    } else if (totalNum < MIN_SPLIT_THRESHOLD * coreNum) {
        usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
    }

    // ---- UB 预算：in(T) + out(T) 双缓冲 + fp32 中间量(tmp) ----
    // half: 2 队列 x2 缓冲 x2B = 8B，另加 fp32 tmp 4B/元素 => 12B/元素
    // float: 2 队列 x2 缓冲 x4B = 16B/元素
    constexpr int64_t ALIGN_ELEM = 16;
    int64_t ubBytesPerElem = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 12 : 16;
    int64_t ubSizePerLoop = static_cast<int64_t>(ubSize) / ubBytesPerElem;
    if (ubSizePerLoop < 1) {
        ubSizePerLoop = 1;
    }
    int64_t ubFactor = FloorAlign(ubSizePerLoop, ALIGN_ELEM);
    if (ubFactor < ALIGN_ELEM) {
        ubFactor = ALIGN_ELEM;
    }
    if (ubFactor > totalNum) {
        ubFactor = CeilAlign(totalNum, ALIGN_ELEM);
    }

    // ---- 每核元素数：按 16 对齐（half 场景 DataCopy 需要）----
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), ALIGN_ELEM);
    if (blockFactor * usedCoreNum > totalNum) {
        // 对齐后若总覆盖超过总量，回退核数保证不越界
        usedCoreNum = CeilDiv(totalNum, blockFactor);
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
