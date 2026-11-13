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
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    // TODO: 实现 Tiling 逻辑
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

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    
    // 获取输入 shape 与 dtype
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    gert::Shape inputShape = EnsureNotScalar(inputShapePtr->GetStorageShape());
    int64_t totalNum = inputShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : TYPE_SIZE;
    
    tiling->totalNum = totalNum;
    // 计算 tilingKey（fp16/bf16 -> MODE_0，fp32 -> MODE_1）
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    // 空 tensor 处理
    if (totalNum == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));       // 32B
    int64_t alignElem = ubBlockSize / typeSize;                          // 32B 对齐的元素数

    // 决定使用的核数：至少 1 个核，最多 coreNum 个核
    int64_t useCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (useCoreNum > coreNum) {
        useCoreNum = coreNum;
    }
    if (useCoreNum < 1) {
        useCoreNum = 1;
    }

    // 每个核处理的元素数（32B 对齐）
    int64_t blockFactor = CeilDiv(totalNum, useCoreNum);
    blockFactor = CeilAlign(blockFactor, alignElem);
    // 修正实际核数（blockFactor 对齐后可能进一步减少核数）
    useCoreNum = CeilDiv(totalNum, blockFactor);
    if (useCoreNum < 1) {
        useCoreNum = 1;
    }

    // 每次 UB 循环处理的元素数：
    //   每份 buffer 需要同时占用 inputQueueX / outputQueueY，每个队列 BUFFER_NUM 份
    //   总 UB 占用 = 2 * BUFFER_NUM * ubFactor * typeSize
    constexpr int64_t BUFFER_NUM = 2;
    int64_t ubFactor = static_cast<int64_t>(ubSize) / (typeSize * BUFFER_NUM * 2);
    ubFactor = FloorAlign(ubFactor, alignElem);
    if (ubFactor < alignElem) {
        ubFactor = alignElem;   // 至少保证一次对齐拷贝
    }
    
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(useCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
