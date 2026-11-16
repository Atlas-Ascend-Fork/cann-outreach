/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// 每个核至少处理这么多个元素，避免小张量拉起过多核、每个核只搬一点点数据
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 与 kernel 侧 tanh.h 中的 BUFFER_NUM 保持一致：输入队列、输出队列各 BUFFER_NUM 块
constexpr int64_t BUFFER_NUM = 2;
// UB 与 GM 之间 DataCopy 的对齐粒度（字节）
constexpr int64_t UB_ALIGN_BYTES = 32;
// 只使用一半 UB，另一半留给系统/框架，避免 UB 超限
constexpr int64_t UB_USABLE_DENOM = 2;

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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // tanh 是逐元素算子，把输入的所有维度乘起来就是需要处理的元素总数
    const ge::DataType dtype = inputDesc->GetDataType();
    const int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;
    const int64_t totalNum = xShape->GetStorageShape().GetShapeSize();

    // 32B 对齐对应的元素个数：float -> 8，half -> 16
    const int64_t alignNum = UB_ALIGN_BYTES / typeSize;

    // 根据输入 dtype 选择 tilingKey，与 op_kernel/tanh.cpp 中的分支一一对应：
    //   fp16/bf16 -> schMode 0（half 计算），fp32 -> schMode 1（float 计算）
    const bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    uint64_t tilingKey = 0;
    if (isFp16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }

    if (totalNum <= 0) {
        // 空张量：只启动一个核，kernel 侧 blockLength 为 0，Process 直接返回
        tiling->totalNum = totalNum;
        tiling->blockFactor = alignNum;
        tiling->ubFactor = alignNum;
        context->SetBlockDim(1);
        context->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    // 多核切分：核数同时受硬件 vector 核数与“每核至少 MIN_SPLIT_THRESHOLD 个元素”限制
    int64_t usedCoreNum = (coreNum > 0) ? coreNum : 1;
    const int64_t minCoreNumByWork = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (usedCoreNum > minCoreNumByWork) {
        usedCoreNum = minCoreNumByWork;
    }

    // 每核元素数向上 32B 对齐，保证每个核在 GM 上的起始地址都是 32B 对齐的
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
    // 按实际切分结果回算真正需要启动的核数（最后一个核可能只分到一小部分）
    usedCoreNum = CeilDiv(totalNum, blockFactor);

    // 每个元素占用的 UB 字节数：
    //   输入队列 BUFFER_NUM 块 + 输出队列 BUFFER_NUM 块，共 2 * BUFFER_NUM 份输入 dtype 数据；
    //   另外 fp32 计算需要 2 块 fp32 临时 buffer（fp16 升精度、中间结果各一块）
    const int64_t bytesPerElem =
        BUFFER_NUM * 2 * typeSize + 2 * static_cast<int64_t>(sizeof(float));
    int64_t ubFactor = FloorAlign(
        static_cast<int64_t>(ubSize / UB_USABLE_DENOM) / bytesPerElem, alignNum);
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }
    // 单核只处理 blockFactor 个元素，UB 块再大也用不上
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling