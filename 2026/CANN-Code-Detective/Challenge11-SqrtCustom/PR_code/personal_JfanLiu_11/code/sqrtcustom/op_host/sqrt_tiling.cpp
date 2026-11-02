/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现：多核连续分段 + 核内按 UB 容量分块
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
using Ops::Base::FloorAlign;

namespace {
constexpr uint32_t UB_RESERVE_BYTES = 1024U;   // 留给 TPipe 管理信息的余量
constexpr int64_t BLOCK_ELEM_GRANULARITY = 256; // 单核任务按 256 元素对齐，避免核间过碎
constexpr int64_t UB_BUFFER_COPIES = 4;         // 双缓冲：2 块输入 + 2 块输出
} // namespace

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    // 1. 平台能力：AIV 核数与单核 UB 容量
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const int64_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "aiv core num is invalid"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ub size is invalid"), return ge::GRAPH_FAILED);

    // 2. 输入 dtype 决定元素宽度、对齐粒度与核函数模板实例
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = 4; // sizeof(float)
        tilingKey = GET_TPL_TILING_KEY(SQRT_SCH_MODE_FP32);
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = 2; // sizeof(half)
        tilingKey = GET_TPL_TILING_KEY(SQRT_SCH_MODE_FP16);
    } else {
        OP_LOGE(context, "only float32 and float16 are supported");
        return ge::GRAPH_FAILED;
    }

    // 3. 输出与输入同形同 dtype，按 ND 各维相乘得到元素总数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const int64_t totalNum = xShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input num must be positive"), return ge::GRAPH_FAILED);

    // 4. 多核切分：每核任务向上对齐到 32B（并粗对齐到 256 元素），尾核自然退化为剩余量
    const int64_t alignNum = 32 / typeSize;
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
    blockFactor = CeilAlign(blockFactor, BLOCK_ELEM_GRANULARITY);
    const int64_t blockDim = CeilDiv(totalNum, blockFactor);

    // 5. 核内切分：按 4 块 buffer 吃满 UB，且不超过单核任务量
    OP_CHECK_IF(static_cast<int64_t>(ubSize) <= static_cast<int64_t>(UB_RESERVE_BYTES),
                OP_LOGE(context, "ub size too small"), return ge::GRAPH_FAILED);
    const int64_t ubCapacity = FloorAlign(
        (static_cast<int64_t>(ubSize) - static_cast<int64_t>(UB_RESERVE_BYTES)) /
            (UB_BUFFER_COPIES * typeSize),
        alignNum);
    OP_CHECK_IF(ubCapacity <= 0, OP_LOGE(context, "ub capacity is zero"), return ge::GRAPH_FAILED);
    const int64_t ubFactor = ubCapacity < blockFactor ? ubCapacity : blockFactor;

    // 6. 落盘 tiling 参数并配置启动核数
    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0U;
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