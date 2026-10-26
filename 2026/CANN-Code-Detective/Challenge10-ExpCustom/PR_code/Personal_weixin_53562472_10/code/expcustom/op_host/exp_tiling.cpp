/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

// 标量（0 维）输入按 1 元素向量处理
static const gert::Shape g_scalarAsVec = {1};
static inline const gert::Shape &AsVector(const gert::Shape &shape)
{
    return shape.GetDimNum() == 0 ? g_scalarAsVec : shape;
}

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息：AIV 核数与单核 UB 容量
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 2. 输入总元素数（ND 各维相乘）
    const gert::StorageShape* storageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    int64_t totalNum = AsVector(storageShape->GetShape()).GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    // 3. dtype 决定元素宽度与 32B 对齐基数，同时决定 tilingKey 模板实例
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    uint64_t tilingKey;
    int64_t typeSize;
    if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
        typeSize = 4;
    } else if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
        typeSize = 2;
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }
    int64_t alignNum = 32 / typeSize; // DataCopy 32B 对齐对应的元素数

    // 4. 多核切分：每核元素数向上对齐到 32B，实际启用核数回收空闲核
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
    int64_t realCoreNum = CeilDiv(totalNum, blockFactor);

    // 5. 核内 UB 切分：单缓冲下每次循环处理的元素数，向下对齐且不超过单核总量
    int64_t ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / typeSize, alignNum);
    ubFactor = ubFactor > blockFactor ? blockFactor : ubFactor;
    ubFactor = ubFactor < alignNum ? alignNum : ubFactor;

    // 6. 落盘 tiling 数据
    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 7. 启动核数、workspace 与 tilingKey
    context->SetBlockDim(realCoreNum);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
