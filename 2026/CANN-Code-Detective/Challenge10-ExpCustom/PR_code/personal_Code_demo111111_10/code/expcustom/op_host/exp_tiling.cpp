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

// 辅助函数：处理标量（0 维）情况
static const gert::Shape g_vec_1_shape = {1};
static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息（UB 大小、核数）
    uint64_t ubSize;
    int64_t coreNum;
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 2. 获取输入 Shape（按照你的要求修改）
    // 注意：TilingContext 返回的是 StorageShape*
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const gert::Shape& inputShape = EnsureNotScalar(inputStorageShape->GetShape());
    int64_t totalNum = inputShape.GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }
    OP_CHECK_IF(totalNum == 0, OP_LOGE(context, "totalNum is 0"), return ge::GRAPH_FAILED);

    // 3. 获取数据类型，计算 typeSize 和对齐基数
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;
    int64_t alignNum = 32 / typeSize; // 32 字节对齐

    // 4. 多核切分逻辑
    int64_t blockLength = CeilDiv(totalNum, coreNum);
    blockLength = CeilAlign(blockLength, alignNum);

    int64_t useCoreNum = CeilDiv(totalNum, blockLength);
    if (useCoreNum > coreNum) {
        useCoreNum = coreNum;
    }

    // 5. UB 循环切分逻辑（Kernel 侧 BUFFER_NUM = 1）
    constexpr int64_t BUFFER_NUM_TILING = 1;
    int64_t tileLength = ubSize / (BUFFER_NUM_TILING * typeSize);
    tileLength = FloorAlign(tileLength, alignNum);
    if (tileLength > blockLength) {
        tileLength = blockLength;
    }
    if (tileLength < alignNum) {
        tileLength = alignNum;
    }
    int64_t tileNum = CeilDiv(blockLength, tileLength);

    // 6. 设置 Tiling 数据
    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    tiling->tileNum = tileNum;

     // 7. 配置启动核数与 Workspace
    context->SetBlockDim(useCoreNum);
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    // 8. 设置 TilingKey（改为 schMode 模式）
    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    } else if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }
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