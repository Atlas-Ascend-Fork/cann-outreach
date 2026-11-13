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

// ---------------------------------------------------------------------------
// Tiling 策略（针对 (8, 2048) 这类小数据量场景）
//   1. 固定 8 个核：16384 个元素 → 每核 2048 个元素，既能并行又不至于每核数据太少；
//   2. 单次 UB 处理 1024 个元素（float32 为 4KB，float16 为 2KB）：
//      每核只需 2 次搬运，配合 kernel 的 BUFFER_NUM=2 双缓冲，
//      MTE2 搬入 / Vector 计算 / MTE3 搬出 可以流水重叠；
//      （若沿用模板默认的 128 个元素，单核要搬 16 次，DataCopy 次数多、单次粒度小）
//   3. workspace 为 0：Tanh 是纯逐元素算子，不需要额外显存。
// ---------------------------------------------------------------------------
constexpr int64_t BLOCK_DIM = 8;     // 启动核数
constexpr int64_t UB_FACTOR = 1024;  // 单次 UB 循环处理元素数上限
constexpr uint32_t WS_SYS_SIZE = 0U; // workspace 大小

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 输入 dtype：决定 tilingKey（float16/bf16 走 0 号模板实例，float 走 1 号）与 32B 对齐粒度
    auto inputDesc = context->GetInputDesc(0);
    int64_t typeSize = 4;
    uint64_t tilingKey;
    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        typeSize = 2;
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
    int64_t alignFactor = 32 / typeSize; // 32B 对齐对应的元素个数

    // 输入总元素数（GetInputShape 返回 StorageShape，需再取 GetShape）
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    if (inputStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t totalNum = inputStorageShape->GetShape().GetShapeSize();

    // 每核元素数：按 8 核均分，向下取整（保证不会读写越界）
    int64_t blockFactor = totalNum / BLOCK_DIM;

    // 单次 UB 处理元素数：默认 1024，但必须能整除每核元素数、且为 32B 对齐的整数倍，
    // 这样 kernel 里不需要任何尾块处理
    int64_t ubFactor = UB_FACTOR;
    if (blockFactor < ubFactor) {
        ubFactor = blockFactor;
    }
    while ((ubFactor > alignFactor) &&
           (((blockFactor % ubFactor) != 0) || ((ubFactor % alignFactor) != 0))) {
        ubFactor >>= 1;
    }
    if ((ubFactor <= 0) || ((ubFactor % alignFactor) != 0)) {
        ubFactor = alignFactor; // 兜底：至少按一个 32B 对齐块处理
    }

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 配置启动核数与 tilingKey
    context->SetBlockDim(static_cast<uint32_t>(BLOCK_DIM));
    context->SetTilingKey(tilingKey);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
