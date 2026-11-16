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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t UB_BLOCK_SIZE = 32;   // UB 一次搬运按 32B 对齐（硬件定值）
constexpr int64_t MAX_UB_ELEMS = 2048;  // 单次 UB 循环元素数上限：9 块缓冲 × 2048 × 4B ≈ 72KB，远低于 UB 容量
constexpr int64_t BUFFER_COUNT = 9;     // UB 占用：输入双缓冲 2 + 输出双缓冲 2 + 计算临时变量 5

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

    // 1. 输入 shape 逐维相乘，得到总元素数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape shape = EnsureNotScalar(xShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalNum *= shape.GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    // 2. 根据 dtype 确定元素大小和 32B 对齐元素数（fp32 → 8，fp16/bf16 → 16）
    auto inputDesc = context->GetInputDesc(0);
    const bool isHalf = (inputDesc != nullptr) &&
                        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16);
    const int64_t typeSize = isHalf ? 2 : TYPE_SIZE;
    const int64_t alignNum = UB_BLOCK_SIZE / typeSize;

    // 3. 多核切分：blockFactor 向上按 32B 对齐；数据量太小只起 1 个核
    const int64_t maxCore = (totalNum < MIN_SPLIT_THRESHOLD) ? 1 : coreNum;
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, maxCore), alignNum);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // 4. UB 内分块：不超过 blockFactor、2048 元素上限和 UB 容量上限，向下对齐
    const int64_t ubCapBySize = static_cast<int64_t>(ubSize) / (BUFFER_COUNT * typeSize);
    int64_t ubFactor = blockFactor;
    if (ubFactor > MAX_UB_ELEMS) {
        ubFactor = MAX_UB_ELEMS;
    }
    if (ubFactor > ubCapBySize) {
        ubFactor = ubCapBySize;
    }
    ubFactor = FloorAlign(ubFactor, alignNum);
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    // 5. 写回 tiling 数据并配置启动核数
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isHalf) {
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