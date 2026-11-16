// /*!
//  * \file relu_tiling.cpp
//  * \brief Relu 算子 Tiling 实现
//  */

// #include "register/op_def_registry.h"
// #include "op_common/log/log.h"
// #include "op_common/op_host/util/math_util.h"
// #include "op_common/op_host/util/platform_util.h"
// #include "../op_kernel/relu_tiling_data.h"
// #include "../op_kernel/relu_tiling_key.h"

// namespace optiling {

// using Ops::Base::CeilDiv;
// using Ops::Base::CeilAlign;
// using Ops::Base::FloorDiv;
// using Ops::Base::FloorAlign;
// using Ops::Base::GetUbBlockSize;

// constexpr uint32_t WS_SYS_SIZE = 0U;
// constexpr int64_t TYPE_SIZE = 4;
// constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// static const gert::Shape g_vec_1_shape = {1};

// static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
//     if (in_shape.GetDimNum() == 0) {
//         return g_vec_1_shape;
//     }
//     return in_shape;
// }

// static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
// {
//     fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
//     OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
//     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
//     coreNum = ascendcPlatform.GetCoreNumAiv();
//     OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
//     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
//     OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
//     return ge::GRAPH_SUCCESS;
// }

// static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
// {
//     size_t* currentWorkspace = context->GetWorkspaceSizes(1);
//     OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
//     currentWorkspace[0] = WS_SYS_SIZE;
//     return ge::GRAPH_SUCCESS;
// }

// static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
// {
//     // TODO: 实现 Tiling 逻辑
//     uint64_t ubSize;
//     int64_t coreNum;
//     OP_CHECK_IF(
//         GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
//         OP_LOGE(context, "GetPlatformInfo error"),
//         return ge::GRAPH_FAILED);

//     OP_CHECK_IF(
//         GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
//         OP_LOGE(context, "GetWorkspaceSize error"),
//         return ge::GRAPH_FAILED);

//     ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
//     OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

//     // TODO: 设置 tiling 数据
//     tiling->totalNum = 0;
//     tiling->blockFactor = 1;
//     tiling->ubFactor = 0;

//     context->SetBlockDim(1);

//     // 根据输入 dtype 选择 tilingKey
//     uint64_t tilingKey;
//     auto inputDesc = context->GetInputDesc(0);
//     if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
//         tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
//     } else {
//         tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
//     }
//     context->SetTilingKey(tilingKey);
//     return ge::GRAPH_SUCCESS;
// }

// static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
// {
//     return ge::GRAPH_SUCCESS;
// }

// struct ReluCompileInfo {};

// IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

// } // namespace optiling
/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;

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
    // 坚决设为 0，防止 Profiling 报错
    currentWorkspace[0] = 0; 
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "error"), return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalNum *= shape.GetDim(i);
    }

    if (totalNum == 0) {
        tiling->totalNum = 0; tiling->blockFactor = 0; tiling->ubFactor = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
        return ge::GRAPH_SUCCESS;
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;

    // 多核切分：严格 32B 对齐
    int64_t alignNum = 32 / typeSize; 
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    blockFactor = CeilAlign(blockFactor, alignNum);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum == 0) usedCoreNum = 1;

    // 单核内 UB 切分：极端保守，缩小 ubFactor 防止越界
    int64_t safeUbSize = static_cast<int64_t>(ubSize) / 2; // 只用一半内存绝对安全
    int64_t maxUbElements = FloorDiv(safeUbSize, typeSize * 2);
    maxUbElements = FloorAlign(maxUbElements, alignNum);
    int64_t ubFactor = (maxUbElements > 0) ? maxUbElements : alignNum;
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling