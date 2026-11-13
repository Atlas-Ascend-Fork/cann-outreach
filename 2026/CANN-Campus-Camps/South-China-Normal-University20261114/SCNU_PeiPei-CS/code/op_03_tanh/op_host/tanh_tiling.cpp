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

const uint32_t TILE_NUM = 8;
// 与 kernel 侧 NsTanh::BUFFER_NUM 保持一致（双缓冲，每块长度要再除以它）
const uint32_t BUFFER_NUM = 2;
// 每核至少处理的元素数：低于这个量级再切多核，调度开销反而盖过收益
const int64_t MIN_LENGTH_PER_CORE = 1024;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    if (tiling == nullptr || x1_shape == nullptr) {
        OP_LOGE(context, "get tiling data or input shape failed");
        return ge::GRAPH_FAILED;
    }

    // 1. 总元素数：shape 各维相乘
    int64_t totalLength = x1_shape->GetStorageShape().GetShapeSize();
    if (totalLength <= 0) {
        OP_LOGE(context, "invalid shape, totalLength = %ld", totalLength);
        return ge::GRAPH_FAILED;
    }

    // 2. 多核切分：按“每核至少 MIN_LENGTH_PER_CORE 个元素”估算并发度，再受 AI Core 数限制
    int64_t coreNum = 1;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        coreNum = platform_ascendc::PlatformAscendC(platformInfo).GetCoreNumAiv();
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    int64_t usedCores = Ops::Base::CeilDiv(totalLength, MIN_LENGTH_PER_CORE);
    if (usedCores <= 0) {
        usedCores = 1;
    }
    if (usedCores > coreNum) {
        usedCores = coreNum;
    }

    // kernel 侧按 totalLength / blockDim 求每核长度，这里保证整除，避免尾部元素被丢掉
    while (usedCores > 1 && (totalLength % usedCores) != 0) {
        --usedCores;
    }
    uint32_t blockDim = static_cast<uint32_t>(usedCores);
    uint32_t blockLength = static_cast<uint32_t>(totalLength / blockDim);

    // 3. 单核内分块：kernel 里每块长度 = blockLength / tileNum / BUFFER_NUM，
    //    同样要保证整除，否则循环次数覆盖不到尾部数据
    uint32_t tileNum = TILE_NUM;
    while (tileNum > 1 && (blockLength % (tileNum * BUFFER_NUM)) != 0) {
        --tileNum;
    }

    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->tileNum = tileNum;
    context->SetBlockDim(blockDim);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
