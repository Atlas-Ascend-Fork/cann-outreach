/*!
 * \file leakyrelu_tiling.cpp
 * \brief Leakyrelu 算子 Tiling 实现
 */

#include <algorithm>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t BLOCK_BYTES = 32;
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

// 读取 negative_slope 属性；属性缺失时退回默认值
static float GetNegativeSlope(gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return LEAKYRELU_DEFAULT_SLOPE;
    }
    const float* slopePtr = attrs->GetFloat(0);
    if (slopePtr == nullptr) {
        slopePtr = attrs->GetAttrPointer<float>(0);
    }
    return (slopePtr == nullptr) ? LEAKYRELU_DEFAULT_SLOPE : *slopePtr;
}

static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
{
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

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const bool isHalf = (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
                         inputDesc->GetDataType() == ge::DT_BF16);
    const int64_t typeSize = isHalf ? 2 : TYPE_SIZE;
    const int64_t alignElems = BLOCK_BYTES / typeSize; // float:8, half:16

    const gert::StorageShape* storageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape shape = EnsureNotScalar(storageShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "invalid input shape"), return ge::GRAPH_FAILED);

    // 1) 多核切分：每核至少 MIN_SPLIT_THRESHOLD 个元素，块长 32B 对齐
    const int64_t splitNum = (totalNum > 0) ? totalNum : 1;
    int64_t usedCoreNum = std::min(coreNum, std::max<int64_t>(1, CeilDiv(splitNum, MIN_SPLIT_THRESHOLD)));
    int64_t blockFactor = CeilAlign(CeilDiv(splitNum, usedCoreNum), alignElems);
    if (blockFactor < alignElems) {
        blockFactor = alignElems;
    }
    usedCoreNum = std::min(coreNum, std::max<int64_t>(1, CeilDiv(splitNum, blockFactor)));

    // 2) UB 切分：输入/输出各 2 块双缓冲，half 分支另需 2 块 float 中间缓冲
    //    按最坏情况估算，保证所有 dtype 都不会超出 UB 容量
    const int64_t bytesPerElem = 2 * BUFFER_NUM * typeSize + 2 * static_cast<int64_t>(sizeof(float));
    int64_t ubCapacity = FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElem, alignElems);
    if (ubCapacity < alignElems) {
        ubCapacity = alignElems;
    }
    const int64_t ubFactor = std::min(blockFactor, ubCapacity);

    // 3) 填充 tiling 数据，并把属性透传给 kernel
    LeakyreluTilingData* tiling = context->GetTilingData<LeakyreluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->negativeSlope = GetNegativeSlope(context);
    tiling->slopeTag = LEAKYRELU_SLOPE_TAG;
    tiling->reserved = 0.0f;

    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    rawTilingData->SetDataSize(sizeof(LeakyreluTilingData));

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据输入 dtype 选择 tilingKey：half -> mode 0，float -> mode 1
    uint64_t tilingKey;
    if (isHalf) {
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
