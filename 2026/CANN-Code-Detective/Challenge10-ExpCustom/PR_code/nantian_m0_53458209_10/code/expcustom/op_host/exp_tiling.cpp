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

    // 获取输入 shape 与 dtype
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);

    int64_t typeSize = TYPE_SIZE;
    ge::DataType dtype = desc->GetDataType();
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        typeSize = 2;
    } else if (dtype != ge::DT_FLOAT) {
        OP_LOGE(context, "unsupported dtype %d", static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }

    // Exp 为逐元素算子，按一维元素总数切分
    int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    int64_t alignNum = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;

    int64_t blockDim = 1;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    if (totalNum > 0) {
        // 多核切分：数据量较小时用单核，避免核间开销
        if (totalNum >= MIN_SPLIT_THRESHOLD) {
            blockDim = std::min<int64_t>(coreNum, CeilDiv(totalNum, alignNum));
            if (blockDim < 1) {
                blockDim = 1;
            }
        }
        blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignNum);

        // UB 切分：输入、输出各一块 buffer，按 32B 向下对齐
        int64_t ubCapacity = FloorAlign(static_cast<int64_t>(ubSize) / (2 * typeSize), alignNum);
        if (ubCapacity < alignNum) {
            ubCapacity = alignNum;
        }
        ubFactor = std::min(blockFactor, ubCapacity);
    } else {
        // 空张量：保证 buffer 长度非 0，kernel 循环次数为 0
        ubFactor = alignNum;
    }

    // TODO: 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
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
