/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现（时延优先）
 *
 * 切分策略（910B3 实测驱动）：
 * 1. 8×2048 输入下，启动 10 个 AIV 核、每核 1 块时延最低：核数再多调度/同步开销
 *    超过并行收益，核数再少单核搬运变长；
 * 2. blockFactor 按 64 元素向上对齐，保证每核起始地址 32B 对齐；
 * 3. ubFactor 限制在 2048 元素以内，保证 UB 占用有上界；
 * 4. 大 shape 时按“每核不超过 MAX_ELEMS_PER_BLOCK 个元素”自动增加核数。
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
constexpr int64_t ALIGN_ELEM = 64;          // 与 kernel 侧一致（32B 对齐 + 完整 vector repeat）
constexpr int64_t TARGET_BLOCKS = 10;       // 小 shape 目标核数（实测最优）
constexpr int64_t MAX_ELEMS_PER_BLOCK = 8192; // 大 shape 时每核最多元素数
constexpr int64_t UB_MAX_ELEMS = 2048;      // 单块元素数上限

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

    // 输入 shape 逐维相乘得到总元素数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& inShape = EnsureNotScalar(xShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < inShape.GetDimNum(); i++) {
        totalNum *= inShape.GetDim(i);
    }
    if (totalNum < 1) {
        totalNum = 1;
    }

    // 目标核数：小 shape 至少 TARGET_BLOCKS 个，大 shape 按每核元素上限增加
    const int64_t unitNum = CeilDiv(totalNum, ALIGN_ELEM);
    int64_t targetBlocks = CeilDiv(totalNum, MAX_ELEMS_PER_BLOCK);
    if (targetBlocks < TARGET_BLOCKS) {
        targetBlocks = TARGET_BLOCKS;
    }
    int64_t blockNum = (coreNum < targetBlocks) ? coreNum : targetBlocks;
    if (blockNum > unitNum) {
        blockNum = unitNum;
    }
    if (blockNum < 1) {
        blockNum = 1;
    }

    // blockFactor 按 64 元素对齐（保证每核起始地址 32B 对齐），再按实际覆盖回算核数
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockNum), ALIGN_ELEM);
    const int64_t actualBlocks = CeilDiv(totalNum, blockFactor);
    if (actualBlocks < blockNum) {
        blockNum = actualBlocks;
    }

    // ubFactor 限制在 UB_MAX_ELEMS 内并 64 对齐；kernel 侧还会再夹紧一次
    int64_t ubFactor = (blockFactor > UB_MAX_ELEMS) ? UB_MAX_ELEMS : blockFactor;
    ubFactor = FloorAlign(ubFactor, ALIGN_ELEM);
    if (ubFactor < ALIGN_ELEM) {
        ubFactor = ALIGN_ELEM;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockNum));

    // 根据输入 dtype 选择 tilingKey（fp16/bf16 -> mode0，fp32 -> mode1）
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
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
