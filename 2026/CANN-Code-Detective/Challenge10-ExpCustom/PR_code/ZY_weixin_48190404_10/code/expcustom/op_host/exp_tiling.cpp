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

static const gert::Shape g_vec_1_shape = {1};
static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 2. 获取输入 Shape（使用你要求的 GetShape 方式）
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const gert::Shape& inputShape = EnsureNotScalar(inputStorageShape->GetShape());
    int64_t totalNum = inputShape.GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }
    OP_CHECK_IF(totalNum == 0, OP_LOGE(context, "totalNum is 0"), return ge::GRAPH_FAILED);

    // 3. 获取 dtype 和 对齐基数
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;
    int64_t alignNum = 32 / typeSize; // 32 字节对齐

    // 4. 多核切分计算 blockFactor
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    blockFactor = CeilAlign(blockFactor, alignNum);

    int64_t useCoreNum = CeilDiv(totalNum, blockFactor);
    if (useCoreNum > coreNum) {
        useCoreNum = coreNum;
    }

    // 5. UB 循环切分计算 ubFactor
    // 模板中 exp.h 的 BUFFER_NUM = 1
    constexpr int64_t BUFFER_NUM_TILING = 1;
    int64_t ubFactor = ubSize / (BUFFER_NUM_TILING * typeSize);
    ubFactor = FloorAlign(ubFactor, alignNum);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    // 6. 写入 Tiling 数据
    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 7. 设置 BlockDim 和 Workspace
    context->SetBlockDim(useCoreNum);
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

        // 8. 设置 TilingKey（根据你的最新 exp_tiling_key.h 修改）
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