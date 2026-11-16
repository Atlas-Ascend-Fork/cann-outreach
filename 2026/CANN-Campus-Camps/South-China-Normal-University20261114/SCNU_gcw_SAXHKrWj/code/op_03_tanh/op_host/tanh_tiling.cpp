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
constexpr int64_t TYPE_SIZE = 4;
// 每核至少处理 2048 个元素：小 shape 下主动少开核，避免每核数据过少时
// 块调度、UB 初始化、DMA 建链等固定开销成为耗时主导
constexpr int64_t MIN_ELEMS_PER_CORE = 2048;
// 单次 UB 循环的元素数上限：每核切成多个小 tile 后，双缓冲才能让
// 下一块的 CopyIn/CopyOut 与当前块的 Compute 重叠，隐藏搬运耗时
constexpr int64_t UB_TILE_ELEMS = 512;
constexpr int64_t BUFFER_NUM = 2;          // 输入/输出队列各开双缓冲
constexpr int64_t UB_RESERVE_SIZE = 8 * 1024;  // UB 预留空间，避免框架/对齐开销导致越界
constexpr int64_t BLOCK_SIZE = 32;         // 一个 UB block 的字节数

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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 获取输入 shape 与数据类型
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    const bool isFp16 = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16);
    const int64_t typeSize = isFp16 ? 2 : TYPE_SIZE;

    // 32 字节对齐：DataCopy 的搬运长度需要按一个 UB block 对齐
    const int64_t alignNum = BLOCK_SIZE / typeSize;

    tiling->totalNum = totalNum;

    int64_t blockFactor = 1;
    int64_t ubFactor = 1;
    int64_t usedCoreNum = 1;
    if (totalNum > 0) {
        // 先按每核最少元素数确定目标核数，小 shape 不再盲目拉满全部核
        int64_t targetCoreNum = CeilDiv(totalNum, MIN_ELEMS_PER_CORE);
        if (targetCoreNum < 1) {
            targetCoreNum = 1;
        }
        if (targetCoreNum > coreNum) {
            targetCoreNum = coreNum;
        }

        // 多核切分：每个核处理 blockFactor 个元素，并向上按 32 字节对齐
        blockFactor = CeilAlign(CeilDiv(totalNum, targetCoreNum), alignNum);
        // 对齐后重新计算实际启动的核数
        usedCoreNum = CeilDiv(totalNum, blockFactor);
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }

        // 单核 UB 内切分：输入队列 BUFFER_NUM 块 + 输出队列 BUFFER_NUM 块
        // + Tanh 高阶接口所需临时空间（按每元素 2 份 float 预留）
        const int64_t perElemBytes =
            BUFFER_NUM * typeSize * 2 + 2 * static_cast<int64_t>(sizeof(float));
        int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVE_SIZE;
        if (ubBudget < 0) {
            ubBudget = 0;
        }
        ubFactor = FloorAlign(ubBudget / perElemBytes, alignNum);
        if (ubFactor < alignNum) {
            ubFactor = alignNum;
        }
        if (ubFactor > blockFactor) {
            ubFactor = blockFactor;
        }
        // 控制单次计算块大小，保证每核至少能切成两块，双缓冲流水才有重叠收益
        if (ubFactor > UB_TILE_ELEMS) {
            ubFactor = UB_TILE_ELEMS;
        }
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey：fp16/bf16 走 half 实例，fp32 走 float 实例
    const uint64_t tilingKey = isFp16 ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
                                      : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
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
