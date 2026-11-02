/*!
 * \file sqrt_tiling.cpp
 * \brief Sqrt 算子 Tiling 实现
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
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t BUFFER_NUM = 2;

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

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
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

    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 注意：TilingContext 返回的是 StorageShape*
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const gert::Shape& inputShape = EnsureNotScalar(inputStorageShape->GetShape());
    int64_t totalNum = inputShape.GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    ge::DataType dtype = inputDesc->GetDataType();
    bool isHalf = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    int64_t typeSize = isHalf ? 2 : TYPE_SIZE;

    uint64_t tilingKey = isHalf
        ? GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(SQRT_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // GetUbBlockSize 是模板函数，必须传 context
    int64_t blockElements = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;
    if (blockElements <= 0) {
        blockElements = 1;
    }

    // 【防越界】按 float 字节数(4)来切割空间，确保 half 升精度内存宽裕
    int64_t ubFactor = static_cast<int64_t>(ubSize) / (2 * BUFFER_NUM * TYPE_SIZE);
    
    // 同样设置一个上限，防止切块太大
    if (ubFactor > 8192) {
        ubFactor = 8192;
    }
    
    ubFactor = FloorAlign(ubFactor, blockElements);

    int64_t alignedBlockFactor = CeilAlign(blockFactor, blockElements);
    if (ubFactor > alignedBlockFactor) {
        ubFactor = alignedBlockFactor;
    }
    if (ubFactor < blockElements) {
        ubFactor = blockElements;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};

IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);

} // namespace optiling