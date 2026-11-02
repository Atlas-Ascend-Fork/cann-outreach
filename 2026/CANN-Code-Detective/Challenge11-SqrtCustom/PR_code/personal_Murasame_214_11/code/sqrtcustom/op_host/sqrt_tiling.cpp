/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现：先定 tile 长度，再按 tile 轮转均摊到各核
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
constexpr uint32_t UB_RESERVE_BYTES = 1024U; // TPipe 管理信息余量
constexpr int64_t TILES_PER_CORE = 2;        // 期望每核至少轮转到 2 个 tile
constexpr int64_t FP32_BYTES = 4;            // fp16 升 fp32 计算的中间宽度
} // namespace

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    // 1. 平台能力
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const int64_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "aiv core num is invalid"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ub size is invalid"), return ge::GRAPH_FAILED);

    // 2. dtype：决定 GM 侧元素宽度与核函数实例
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = 4; // sizeof(float)
        tilingKey = GET_TPL_TILING_KEY(SQRT_CALC_MODE_FP32);
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = 2; // sizeof(half)
        tilingKey = GET_TPL_TILING_KEY(SQRT_CALC_MODE_FP16_UPCAST);
    } else {
        OP_LOGE(context, "only float32 and float16 are supported");
        return ge::GRAPH_FAILED;
    }

    // 3. 元素总数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const int64_t totalNum = xShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input num must be positive"), return ge::GRAPH_FAILED);

    // 4. 第一层：按 UB 容量求单 tile 上限。单缓冲下 UB 中同时存在输入副本、输出副本
    //    以及 fp16 路径的 fp32 中间结果，故每个元素占 (2 * typeSize + 4) 字节。
    const int64_t alignNum = 32 / typeSize;
    OP_CHECK_IF(static_cast<int64_t>(ubSize) <= static_cast<int64_t>(UB_RESERVE_BYTES),
                OP_LOGE(context, "ub size too small"), return ge::GRAPH_FAILED);
    const int64_t bytesPerElement = 2 * typeSize + FP32_BYTES;
    const int64_t ubTileCapacity =
        FloorAlign((static_cast<int64_t>(ubSize) - static_cast<int64_t>(UB_RESERVE_BYTES)) / bytesPerElement, alignNum);
    OP_CHECK_IF(ubTileCapacity <= 0, OP_LOGE(context, "ub tile capacity is zero"), return ge::GRAPH_FAILED);

    // 5. 第二层：为了让每个核都能轮到若干 tile，按 (总元素 / 核数 / 每核 tile 数) 再收一次上限
    const int64_t balancedTile = CeilAlign(CeilDiv(CeilDiv(totalNum, coreNum), TILES_PER_CORE), alignNum);
    const int64_t tileLength = balancedTile < ubTileCapacity ? balancedTile : ubTileCapacity;
    OP_CHECK_IF(tileLength <= 0, OP_LOGE(context, "tile length is zero"), return ge::GRAPH_FAILED);

    // 6. tile 总数决定实际起核数，单核最多轮到的元素数作为 blockFactor 记录
    const int64_t tileCount = CeilDiv(totalNum, tileLength);
    const int64_t blockDim = tileCount < coreNum ? tileCount : coreNum;
    const int64_t blockFactor = CeilDiv(tileCount, blockDim) * tileLength;

    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = tileLength;

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