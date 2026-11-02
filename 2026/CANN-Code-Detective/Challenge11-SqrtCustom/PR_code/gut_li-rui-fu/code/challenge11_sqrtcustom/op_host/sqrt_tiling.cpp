/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现
 *
 * 多核切分：总元素数按 AIV 核数均分，blockFactor 向 32B 对齐（保证各核 GM 起始地址对齐）；
 * UB 切分：UB 容量按 dtype + 双缓冲份数（输入 2 块 + 输出 2 块）均分。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;
// UB 中同时驻留的 tensor 份数：输入队列 2 块（双缓冲）+ 输出队列 2 块（双缓冲）
constexpr int64_t UB_TENSOR_NUM = 4;

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
    // 即使 workspace 为 0，也必须声明 slot，否则运行时不会为 kernel 传 workspace 指针
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 取输入元素总数与 dtype；同时校验输出 shape 与输入一致
static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "invalid input shape size: %ld", totalNum), return ge::GRAPH_FAILED);

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const int64_t outputNum = outputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(
        outputNum != totalNum, OP_LOGE(context, "Sqrt: shape mismatch, x=%ld, y=%ld", totalNum, outputNum),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Sqrt: unsupported dtype %d, only FLOAT/FLOAT16 are supported", static_cast<int32_t>(dataType)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetShapeAndDtype(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndDtype error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
    // DMA 最小访问粒度 32B 折算成元素个数：fp32 → 8，fp16 → 16
    const int64_t blockAlign = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;
    OP_CHECK_IF(blockAlign <= 0, OP_LOGE(context, "invalid block align"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;

    // 空 tensor：不启动循环，留 1 个核兜底，仍按 dtype 设置 tilingKey 保证 kernel 可正常下发
    if (totalNum == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = blockAlign;
        context->SetBlockDim(1);
        context->SetTilingKey(
            GET_TPL_TILING_KEY(
                (dataType == ge::DT_FLOAT16) ? SQRT_TPL_SCH_MODE_0 : SQRT_TPL_SCH_MODE_1));
        return ge::GRAPH_SUCCESS;
    }

    // 核数按数据量收敛：每核至少 MIN_ELEMS_PER_CORE 个元素。
    // 小数据量下把 40 个 AIV 全开并不划算——实测 (8,2048) fp32 从 40 核降到 8 核，
    // kernel 耗时 4.66us -> 2.06us，省下的正是每块的启动/流水初始化开销。
    constexpr int64_t MIN_ELEMS_PER_CORE = 2048;
    int64_t coreLimit = CeilDiv(totalNum, MIN_ELEMS_PER_CORE);
    if (coreLimit < 1) {
        coreLimit = 1;
    }
    if (coreLimit > coreNum) {
        coreLimit = coreNum;
    }

    // 多核切分：blockFactor 向 32B 对齐，避免尾块 DataCopyPad 写出越过本核区间
    tiling->blockFactor = CeilAlign(CeilDiv(totalNum, coreLimit), blockAlign);
    const int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // UB 切分：UB 字节数 / dtype 大小 / 同时驻留的 tensor 份数，再向 32B 对齐
    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, UB_TENSOR_NUM), blockAlign);
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubSize too small, ubFactor is 0"), return ge::GRAPH_FAILED);
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey（对应 sqrt.cpp 中 sqrt<schMode> 的模板实参）
    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};

IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);

} // namespace optiling
