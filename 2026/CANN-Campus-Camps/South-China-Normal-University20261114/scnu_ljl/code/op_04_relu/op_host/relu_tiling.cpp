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

constexpr uint32_t WS_SYS_SIZE = 0U;

// ==================== 可调参数 ====================
constexpr int64_t UB_ALIGN        = 32;     // UB 对齐粒度（字节）
constexpr int64_t MIN_CORE_DIM    = 8;      // 小数据用的核数
constexpr int64_t MAX_CORE_DIM    = 16;     // 大数据用的核数
constexpr int64_t CORE_SWITCH_NUM = 65536;  // 元素数超过此值则用 MAX_CORE_DIM
// =================================================

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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---------- 总元素数 ----------
    int64_t totalNum = 1;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& inShape = xShape->GetStorageShape();
    for (size_t i = 0; i < inShape.GetDimNum(); ++i) {
        totalNum *= inShape.GetDim(i);
    }

    // ---------- 数据类型字长 ----------
    int64_t typeSize = 4;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr) {
        int64_t sz = static_cast<int64_t>(ge::GetSizeByDataType(inputDesc->GetDataType()));
        if (sz > 0) {
            typeSize = sz;
        }
    }

    int64_t alignElem = UB_ALIGN / typeSize;
    if (alignElem <= 0) {
        alignElem = 1;
    }

    // ---------- 自适应选核数 ----------
    int64_t usedCoreNum = (totalNum > CORE_SWITCH_NUM) ? MAX_CORE_DIM : MIN_CORE_DIM;
    // 数据太小时不浪费核
    int64_t needCore = (totalNum + alignElem - 1) / alignElem;
    if (usedCoreNum > needCore) {
        usedCoreNum = needCore;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    // ---------- 每核元素数（32B 对齐） ----------
    int64_t perCore = (totalNum + usedCoreNum - 1) / usedCoreNum;
    perCore = ((perCore + alignElem - 1) / alignElem) * alignElem;
    if (perCore <= 0) {
        perCore = alignElem;
    }

    // ---------- 单次 UB 搬运上限（双缓冲占 4 份） ----------
    int64_t ubAvail = static_cast<int64_t>(ubSize) / 4;
    int64_t maxUbNum = (ubAvail / typeSize / alignElem) * alignElem;
    if (maxUbNum <= 0) {
        maxUbNum = alignElem;
    }

    int64_t ubFactor = perCore;
    if (ubFactor > maxUbNum) {
        ubFactor = maxUbNum;
    }

    // ---------- 写回 tiling ----------
    tiling->totalNum = totalNum;
    tiling->blockFactor = perCore;
    tiling->ubFactor = ubFactor;

    // ---------- 配置启动核数 ----------
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // ---------- tilingKey ----------
    uint64_t tilingKey;
    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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