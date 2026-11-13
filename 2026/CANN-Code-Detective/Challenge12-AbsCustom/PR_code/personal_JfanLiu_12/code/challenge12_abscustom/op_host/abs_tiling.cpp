/*!
 * \file abs_tiling.cpp
 * \brief Abs 算子 Tiling 实现：多核连续分段 + 核内按 UB 容量分块
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/abs_tiling_data.h"
#include "../op_kernel/abs_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

namespace {
constexpr int64_t TILE_NUM = 16;              // 核内默认分块数
constexpr int64_t UB_RESERVE_BYTES = 1024;    // 留给 TPipe 管理信息的余量
constexpr int64_t UB_BUFFER_COPIES = 4;       // 双缓冲：2 块输入 + 2 块输出
} // namespace

static ge::graphStatus AbsTilingFunc(gert::TilingContext* context)
{
    // 1. 输入 dtype 决定元素宽度与核函数模板实例
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    uint32_t dtKey = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtKey);

    int64_t typeSize = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = 4; // sizeof(float)
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = 2; // sizeof(half)
    } else {
        OP_LOGE(context, "Abs only supports float32 and float16");
        return ge::GRAPH_FAILED;
    }

    // 2. 平台能力：AIV 核数与单核 UB 容量
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const int64_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "aiv core num is invalid"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ub size is invalid"), return ge::GRAPH_FAILED);

    // 3. 输出与输入同形，按 ND 各维相乘得到元素总数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const int64_t totalNum = xShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input num must be positive"), return ge::GRAPH_FAILED);

    // 4. 多核切分：每核任务向上对齐到 32B，尾核自然退化为剩余量
    const int64_t alignNum = 32 / typeSize;
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
    const int64_t blockDim = CeilDiv(totalNum, blockFactor);

    // 5. 核内切分：默认 16 块，同时保证单块不撑爆 UB、且不小于 32B
    const int64_t ubCapacity = FloorAlign(
        (static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES) / (UB_BUFFER_COPIES * typeSize), alignNum);
    OP_CHECK_IF(ubCapacity <= 0, OP_LOGE(context, "ub capacity is zero"), return ge::GRAPH_FAILED);
    int64_t tileNum = TILE_NUM;
    const int64_t minTileNum = CeilDiv(blockFactor, ubCapacity);
    int64_t maxTileNum = blockFactor / alignNum;
    maxTileNum = maxTileNum < 1 ? 1 : maxTileNum;
    tileNum = tileNum < minTileNum ? minTileNum : tileNum;
    tileNum = tileNum > maxTileNum ? maxTileNum : tileNum;
    tileNum = tileNum < 1 ? 1 : tileNum;

    // 6. 落盘 tiling 参数并配置启动核数
    AbsTilingData* tiling = context->GetTilingData<AbsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->tileNum = tileNum;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAbs([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AbsCompileInfo {};

IMPL_OP_OPTILING(Abs).Tiling(AbsTilingFunc).TilingParse<AbsCompileInfo>(TilingParseForAbs);

} // namespace optiling
