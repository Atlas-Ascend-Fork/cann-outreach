/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现（版本 E：实测驱动的自适应策略）
 *
 * 实测结论（平台耗时，两个测试点分别为 (8,2048) 64KB 与 (2048,2048) 16MB）：
 *   1. 小 shape 是启动开销主导：核数从 16 增到 20/32、每核块长从 4KB 降到 2~3.3KB 后，
 *      耗时从 5.70µs 恶化到 10.19~13.04µs。=> 小 shape 必须少核、大块。
 *   2. 大 shape 的 tile 尺寸有最优值：46KB tile（19 tiles）26.86µs，
 *      32KB tile（26 tiles）23.50µs，后者快约 12%。=> 大 shape 用 32KB tile。
 *
 * 因此本版本采用「分档」策略：
 *   - 分核规则沿用实测最优的那一档（阈值 1024 元素），保证小 shape 的实测性能不被改动；
 *   - 仅对单核数据量超过 tile 上限的情况把 tile 收敛到 32KB，改善大 shape。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// 单核最小数据量（元素）。该值是实测最优档：小 shape 用较少核、每核 4KB，
// 避免核数过多带来的启动偏斜与小包 DMA 开销。
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// UB 上 DataCopy 的对齐粒度（字节）
constexpr int64_t UB_BLOCK_BYTES = 32;
// 单个 tile 的字节数上限。实测 32KB 优于 46KB（大 shape 快约 12%）：
// 更小的 tile 让 DoubleBuffer 的搬入/计算/搬出重叠更充分。
constexpr int64_t TILE_BYTES_CAP = 32 * 1024;
// 【E2 实验参数】小 shape 进一步减少核数：总数据量不超过 128KB 时，目标每核 8KB
constexpr int64_t SMALL_TOTAL_BYTES = 256 * 1024;
constexpr int64_t SMALL_CHUNK_BYTES = 64 * 1024;
// UB 预算系数：预留少量 UB 给运行时/对齐开销
constexpr int64_t UB_BUDGET_NUM = 15;
constexpr int64_t UB_BUDGET_DEN = 16;
// DoubleBuffer 深度：输入队列、输出队列各 2 块 buffer
constexpr int64_t BUFFER_NUM = 2;

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

    // ---------------- 1. 解析输入信息 ----------------
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& xShape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = xShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    // float16 / bf16 为 2 字节，其余（float32 等）按 4 字节处理
    const int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;
    const int64_t alignNum = UB_BLOCK_BYTES / typeSize;

    // ---------------- 2. 启动核数与每核数据量 ----------------
    int64_t usedCoreNum = 1;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    if (totalNum > 0) {
        // 数据量小时按阈值减少启动核数：小 shape 下核越多、每核块越碎，实测越慢
        int64_t chunkNum = MIN_SPLIT_THRESHOLD;
        if (totalNum * typeSize <= SMALL_TOTAL_BYTES) {
            chunkNum = SMALL_CHUNK_BYTES / typeSize;
        }
        const int64_t needCoreNum = CeilDiv(totalNum, chunkNum);
        usedCoreNum = needCoreNum < coreNum ? needCoreNum : coreNum;
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }
        // 每核负责的元素个数按 32B block 对齐
        blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
        // 对齐后重算实际需要的核数，末尾核不空跑
        usedCoreNum = CeilDiv(totalNum, blockFactor);

        // ---------------- 3. 单 tile 元素数 ----------------
        // 同时受两个约束：UB 预算（DoubleBuffer 下共 2*BUFFER_NUM 块数据）、tile 字节上限
        const int64_t ubBudget = static_cast<int64_t>(ubSize) * UB_BUDGET_NUM / UB_BUDGET_DEN;
        int64_t tileFactor = ubBudget / (2 * BUFFER_NUM * typeSize);
        const int64_t tileCapNum = TILE_BYTES_CAP / typeSize;
        if (tileFactor > tileCapNum) {
            tileFactor = tileCapNum;
        }
        tileFactor = FloorAlign(tileFactor, alignNum);
        if (tileFactor < alignNum) {
            tileFactor = alignNum;
        }
        // tile 不需要超过单核负责的数据量
        ubFactor = tileFactor > blockFactor ? blockFactor : tileFactor;
        if (ubFactor < 1) {
            ubFactor = 1;
        }
    }

    // ---------------- 4. 写回 tiling 数据 ----------------
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
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
