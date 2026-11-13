/*!
 * \file leakyrelu_tiling.cpp
 * \brief Leakyrelu 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t QUEUE_BUFFER_NUM = 2;
constexpr int64_t QUEUE_COUNT = 2;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "AIV core number is zero"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "UB size is zero"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
{
    // 1. 获取输入Shape和数据类型。
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "Input tensor is empty"), return ge::GRAPH_FAILED);

    const auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const ge::DataType inputDtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    if (inputDtype == ge::DT_FLOAT) {
        typeSize = 4;
    } else if (inputDtype == ge::DT_FLOAT16) {
        typeSize = 2;
    } else {
        OP_LOGE(context, "Leakyrelu only supports float32 and float16");
        return ge::GRAPH_FAILED;
    }

    // 2. 从op_def注册的属性中读取negative_slope。
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* negativeSlopePtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, negativeSlopePtr);
    const float negativeSlope = *negativeSlopePtr;

    // 3. 获取昇腾平台的AIV核数和UB容量。
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. 根据总元素数进行多核切分。
    // 每个核至少处理约MIN_SPLIT_THRESHOLD个元素，避免小Tensor启动过多核。
    int64_t blockDim = std::min<int64_t>(coreNum, CeilDiv(totalNum, MIN_SPLIT_THRESHOLD));
    if (blockDim < 1) {
        blockDim = 1;
    }

    // DataCopy要求搬运长度按UB数据块对齐。910B的UB数据块通常为32 Byte。
    const int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    const int64_t alignElements = std::max<int64_t>(1, ubBlockSize / typeSize);
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignElements);

    // 每个核同时维护输入、输出两个队列，每个队列使用双缓冲。
    // 为队列预留的总UB空间不超过平台UB容量。
    const int64_t maxUbElements =
        static_cast<int64_t>(ubSize / (QUEUE_BUFFER_NUM * QUEUE_COUNT * typeSize));
    const int64_t ubFactor = FloorAlign(std::min(blockFactor, maxUbElements), alignElements);
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "UB is insufficient for one aligned tile"),
                return ge::GRAPH_FAILED);

    // 5. 将切分参数和属性写入TilingData，供Kernel侧使用。
    LeakyreluTilingData* tiling = context->GetTilingData<LeakyreluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->negativeSlope = negativeSlope;
    tiling->reserved = 0;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 6. 根据输入dtype选择Kernel模板分支：0对应float16，1对应float32。
    uint64_t tilingKey = 0;
    if (inputDtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLeakyrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LeakyreluCompileInfo {};

IMPL_OP_OPTILING(Leakyrelu).Tiling(LeakyreluTilingFunc).TilingParse<LeakyreluCompileInfo>(TilingParseForLeakyrelu);

} // namespace optiling
