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

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

static const gert::Shape SCALAR_SHAPE = {1};

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 ? SCALAR_SHAPE : shape;
}

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);

    const int64_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "AI Vector core number is invalid"),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "UB size is zero"), return ge::GRAPH_FAILED);

    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    int64_t totalNum = EnsureNotScalar(inputStorageShape->GetShape()).GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input has no elements"), return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = sizeof(float);
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = sizeof(int16_t);
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    } else {
        OP_LOGE(context, "unsupported input dtype");
        return ge::GRAPH_FAILED;
    }

    const int64_t alignNum = 32 / typeSize;
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
    int64_t useCoreNum = CeilDiv(totalNum, blockFactor);
    if (useCoreNum > coreNum) {
        useCoreNum = coreNum;
    }

    constexpr int64_t BUFFER_NUM = 1;
    constexpr int64_t TENSOR_COUNT = 2;
    int64_t ubFactor = FloorAlign(
        static_cast<int64_t>(ubSize) / (BUFFER_NUM * TENSOR_COUNT * typeSize), alignNum);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    // DataCopyParams::blockLen is uint16_t and stores the byte count.
    const int64_t maxCopyElements = FloorAlign(65535 / typeSize, alignNum);
    if (ubFactor > maxCopyElements) {
        ubFactor = maxCopyElements;
    }
    OP_CHECK_IF(ubFactor < alignNum, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(useCoreNum);
    context->SetTilingKey(tilingKey);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
