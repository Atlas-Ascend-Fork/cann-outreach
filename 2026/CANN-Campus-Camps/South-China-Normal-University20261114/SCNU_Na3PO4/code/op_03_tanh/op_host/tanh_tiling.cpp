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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
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

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 1. 计算输入总元素数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& inShape = xShape->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < inShape.GetDimNum(); i++) {
        totalNum *= inShape.GetDim(i);
    }

    // 2. 取输入 dtype，确定元素字节数（kernel 中 fp16 升 fp32 计算，UB 预算按 fp32 临时变量算）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : TYPE_SIZE;

    // 3. 32B 对齐（DMA 搬运最小粒度）对应的元素个数
    const int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    const int64_t alignElems = ubBlockSize / typeSize;

    if (totalNum <= 0) {
        // 空 tensor：不启动有效计算，避免除零
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
    } else {
        // 4. 多核切分：均分到 coreNum 个 AIV，blockFactor 按 32B 向上对齐，
        //    这样每个核的 GM 起始地址都是 32B 对齐，相邻核不会互相覆盖
        tiling->totalNum = totalNum;
        tiling->blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElems);
        int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

        // 5. UB 切分：每个元素的 UB 占用 =
        //    输入双缓冲 2 份 + 输出双缓冲 2 份（各 sizeof(T)） + 4 份 float32 临时变量
        const int64_t ubBytesPerElem = 4 * typeSize + 4 * TYPE_SIZE;
        int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), ubBytesPerElem), alignElems);
        if (ubFactor < alignElems) {
            ubFactor = alignElems;
        }
        tiling->ubFactor = ubFactor;

        context->SetBlockDim(usedCoreNum);
    }

    // 6. 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDescForKey = context->GetInputDesc(0);
    if (inputDescForKey != nullptr &&
        (inputDescForKey->GetDataType() == ge::DT_FLOAT16 || inputDescForKey->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
