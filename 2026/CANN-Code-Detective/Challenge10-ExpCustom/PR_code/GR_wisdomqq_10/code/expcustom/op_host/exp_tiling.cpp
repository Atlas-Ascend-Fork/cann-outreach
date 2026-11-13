/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现（多核切分 + 核内 UB 分块）
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

constexpr uint32_t WS_SYS_SIZE = 0U;
// 单次 UB 循环元素数上限：fp32 下 2048 元素 = 8KB/队列，双队列双缓冲共 32KB，远小于 192KB UB
constexpr int64_t UB_FACTOR_MAX = 2048;
// DataCopy / UB 分配的最小对齐单元（字节）
constexpr int64_t ALIGN_BYTES = 32;
// 单个元素在 UB 中的份数：输入队列 + 输出队列，各双缓冲
constexpr int64_t UB_USAGE_FACTOR = 4;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetUbSize(gert::TilingContext* context, uint64_t& ubSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetCoreNum(gert::TilingContext* context, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    if (coreNum <= 0) {
        OP_LOGE(context, "coreNum is %ld", coreNum);
        coreNum = 1;
    }
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
    OP_CHECK_IF(
        GetUbSize(context, ubSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetUbSize error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 输入 dtype 决定模板参数 schMode（同时决定 32B 对齐的元素粒度），
    // schMode 取值必须与 op_kernel/exp_tiling_key.h 中 1bit 位宽的取值域一致
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const uint32_t schMode = (dtype == ge::DT_FLOAT16) ? EXP_TPL_SCH_HALF : EXP_TPL_SCH_FLOAT;
    // 由模板参数推导并下发 tilingKey，保证 host 下发的 key 与 kernel 侧实例一一对应
    ASCENDC_TPL_SEL_PARAM(context, schMode);
    const int64_t elemBytes = (dtype == ge::DT_FLOAT16) ? 2 : 4;
    const int64_t alignElems = ALIGN_BYTES / elemBytes;  // 32B 对应的元素数（fp32=8, fp16=16）

    // 输入 shape 逐维相乘得到总元素数
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape shape = EnsureNotScalar(x_shape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalNum *= shape.GetDim(i);
    }

    // 空 tensor 防护：不下发任何计算，kernel 侧 blockFactor<=0 时 Process 直接返回
    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = alignElems;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // 多核切分：按平台实际 AIV 核数均分，blockFactor 向上对齐到 32B 粒度
    int64_t coreNum = 1;
    OP_CHECK_IF(
        GetCoreNum(context, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetCoreNum error"),
        return ge::GRAPH_FAILED);

    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElems);
    // 小张量按实际元素数缩减核数，避免大量空核
    int64_t blockDim = CeilDiv(totalNum, blockFactor);
    if (blockDim < 1) {
        blockDim = 1;
    }
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }
    // 用实际启用的核数回算每核窗口，保证 blockDim * blockFactor >= totalNum
    blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignElems);

    // 核内 UB 分块：三重约束取最小 —— UB 容量、单次上限、本核长度
    const int64_t ubLimitByMem = static_cast<int64_t>(ubSize) / (UB_USAGE_FACTOR * elemBytes);
    int64_t ubFactor = UB_FACTOR_MAX;
    if (ubFactor > ubLimitByMem) {
        ubFactor = ubLimitByMem;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    // 32B 对齐后仍需保证下限，避免 ubFactor 为 0 时 kernel 侧除零
    ubFactor = (ubFactor / alignElems) * alignElems;
    if (ubFactor < alignElems) {
        ubFactor = alignElems;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
