// /*!
//  * \file tanh_tiling.cpp
//  * \brief Tanh 算子 Tiling 实现
//  */

// #include "register/op_def_registry.h"
// #include "op_common/log/log.h"
// #include "op_common/op_host/util/math_util.h"
// #include "op_common/op_host/util/platform_util.h"
// #include "../op_kernel/tanh_tiling_data.h"
// #include "../op_kernel/tanh_tiling_key.h"

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

// static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
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

//     TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
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
//         tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
//     } else {
//         tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
//     }
//     context->SetTilingKey(tilingKey);
//     return ge::GRAPH_SUCCESS;
// }

// static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
// {
//     return ge::GRAPH_SUCCESS;
// }

// struct TanhCompileInfo {};

// IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

// } // namespace optiling
/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h" // 必须加上这个头文件
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

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
    // 【修复 1】申请系统 Workspace，防止底层 API 报 507035
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize(); 
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

    // 1. 获取输入 shape，计算总元素数
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalNum *= shape.GetDim(i);
    }

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1));
        return ge::GRAPH_SUCCESS;
    }

    // 2. 确定数据类型和元素字节数
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;

    // 3. 多核切分
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    int64_t alignNum = 32 / typeSize;  // 32B 对齐元素数
    blockFactor = CeilAlign(blockFactor, alignNum);
    if (blockFactor * coreNum > totalNum) {
        blockFactor = FloorAlign(totalNum, alignNum);
    }
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }

    // 4. 单核内 UB 切分
    // 【修复 2】严谨评估需要的 UB 内存：由于 float16 会转 float32 计算，最多需要 6 份空间
    int64_t calcSize = sizeof(float); 
    int64_t maxUbElements = FloorDiv(static_cast<int64_t>(ubSize), calcSize * 6);
    maxUbElements = FloorAlign(maxUbElements, alignNum);
    
    // 【修复 3】如果 maxUbElements 算出来太大，强制限制单次处理量，防止异常
    int64_t ubFactor = (maxUbElements > 0) ? maxUbElements : alignNum;
    if (ubFactor > 4096) { // 对于单核数据，4096 足够，限制一下绝对安全
        ubFactor = 4096;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    // 5. 写入 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // 6. 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
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
