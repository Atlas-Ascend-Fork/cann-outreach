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
#include <algorithm>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

constexpr int64_t kHalfTileCap = 16*1024;//8192;
constexpr int64_t kFloatTileCap = 8*1024;//4096;
constexpr int64_t kUbGuardBytes = 16 * 1024;
constexpr size_t kWorkspaceSlots = 1;
constexpr int64_t kDmaBlockBytes = 32;
constexpr int64_t kVectorPackAlign = 64;
constexpr int64_t kSmallCoreGrain = 512;
constexpr int64_t kLargeCoreGrain = 768;
constexpr int64_t kLargeSplitPoint = 8192;
constexpr int64_t kMiddleCoreGrain = 1024;
constexpr int64_t kMiddleSplitLimit = 262144;
constexpr int64_t kX5SingleTileLimit = 262144;

static int64_t EstimateUbStride(ge::DataType dtype)
{
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    return 2 * typeSize;
}
static int64_t RoundUpDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}
static int64_t RoundUpTo(int64_t value, int64_t align)
{
    return RoundUpDiv(value, align) * align;
}
static int64_t RoundDownTo(int64_t value, int64_t align)
{
    return value / align * align;
}
static int64_t ChooseVectorAlign(ge::DataType dtype)
{
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    int64_t coreAlign = kDmaBlockBytes / typeSize;
    return std::max(coreAlign, kVectorPackAlign);
}
static int64_t PickCoreGrain(int64_t elemCount)
{
    if (elemCount > kLargeSplitPoint && elemCount <= kMiddleSplitLimit) {
        return kMiddleCoreGrain;
    }
    return elemCount > kLargeSplitPoint ? kLargeCoreGrain : kSmallCoreGrain;
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
static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
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
    if (coreNum <= 0) {
        coreNum = 1;
    }
    if (ubSize <= kUbGuardBytes) {
        return ge::GRAPH_FAILED;
    }
    
    const gert::Tensor *xTensor = context->GetRequiredInputTensor(0);
    if (xTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType xDtype = xTensor->GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    int64_t elemCount = static_cast<int64_t>(xTensor->GetShapeSize());
    auto shapeView = context->GetInputShape(0);
    if (shapeView != nullptr) {
        auto storageView = shapeView->GetStorageShape();
        elemCount = storageView.GetDimNum() == 0 ? 1 : static_cast<int64_t>(storageView.GetShapeSize());
    }
    if (elemCount < 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t packAlign = ChooseVectorAlign(xDtype);
    int64_t tileAlignHint = (xDtype == ge::DT_FLOAT) ? kFloatTileCap : kHalfTileCap;
    int64_t plannedCores = 1;
    if (elemCount > 0) {
        int64_t coreGrain = PickCoreGrain(elemCount);
        plannedCores = std::min(coreNum, std::max<int64_t>(1, RoundUpDiv(elemCount, coreGrain)));
    }
    int64_t rawCoreSpan = (elemCount == 0) ? 0 : RoundUpDiv(elemCount, plannedCores);
    int64_t coreAlign = (rawCoreSpan > tileAlignHint) ? tileAlignHint : packAlign;
    int64_t coreSpan = (elemCount == 0) ? 0 : RoundUpTo(rawCoreSpan, coreAlign);
    int64_t launchCores = (elemCount == 0) ? 1 : RoundUpDiv(elemCount, coreSpan);
    launchCores = std::max<int64_t>(1, std::min(coreNum, launchCores));
    
    int64_t ubBytesPerElem = EstimateUbStride(xDtype);
    int64_t ubTileLimit = RoundDownTo((static_cast<int64_t>(ubSize) - kUbGuardBytes) / ubBytesPerElem,
                                      packAlign);
    int64_t typeTileCap = (xDtype == ge::DT_FLOAT) ? kFloatTileCap : kHalfTileCap;
    int64_t tileSpan = std::max<int64_t>(packAlign, std::min<int64_t>(typeTileCap, ubTileLimit));
    if (coreSpan > 0) {
        tileSpan = std::min(tileSpan, RoundUpTo(coreSpan, packAlign));
        tileSpan = std::max<int64_t>(packAlign, RoundDownTo(tileSpan, packAlign));
    }
 
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    // TODO: 设置 tiling 数据
    tiling->total_elems = elemCount;
    tiling->core_elems = coreSpan;
    tiling->tile_elems = tileSpan;

    context->SetBlockDim(static_cast<uint32_t>(launchCores));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc);

} // namespace optiling
