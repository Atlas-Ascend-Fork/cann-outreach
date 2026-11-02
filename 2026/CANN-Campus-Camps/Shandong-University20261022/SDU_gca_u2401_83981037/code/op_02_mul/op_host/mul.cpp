#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *x = context->GetRequiredInputTensor(0);
    const gert::Tensor *y = context->GetRequiredInputTensor(1);
    if (x == nullptr || y == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = x->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        dtype != y->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    const int64_t length = x->GetShapeSize();
    if (length < 0 || length != y->GetShapeSize() ||
        static_cast<uint64_t>(length) > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    constexpr uint64_t RESERVED_UB = 4096;
    constexpr uint64_t BUFFER_COUNT = 6;  // x、y、z 三个队列，各两个 buffer。
    if (coreNum <= 0 || ubSize < RESERVED_UB + BUFFER_COUNT * 32) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t elementSize = (dtype == ge::DT_FLOAT) ? 4 : 2;
    const uint32_t alignNum = 32 / elementSize;
    const uint64_t totalBlocks =
        (static_cast<uint64_t>(length) + alignNum - 1) / alignNum;
    const uint32_t blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(coreNum, std::max<uint64_t>(1, totalBlocks)));

    // 最大核工作量，以及 UB 能容纳的 tile 容量，均按 32 字节对齐。
    const uint64_t maxCoreLength =
        std::max<uint64_t>(1, (totalBlocks + blockDim - 1) / blockDim) * alignNum;
    const uint64_t ubTileLength =
        ((ubSize - RESERVED_UB) / (BUFFER_COUNT * 32)) * alignNum;
    const uint32_t tileLength = static_cast<uint32_t>(
        std::min<uint64_t>(8192, std::min(maxCoreLength, ubTileLength)));

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = static_cast<uint32_t>(length);
    tiling->tileLength = tileLength;
    const uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x = context->GetInputShape(0);
    const gert::Shape *y = context->GetInputShape(1);
    gert::Shape *z = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr || z == nullptr ||
        x->GetDimNum() != y->GetDimNum()) {
        return GRAPH_FAILED;
    }
    for (size_t i = 0; i < x->GetDimNum(); ++i) {
        if (x->GetDim(i) != y->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }
    *z = *x;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto dtype = context->GetInputDataType(0);
    if ((dtype != DT_FLOAT && dtype != DT_FLOAT16) ||
        dtype != context->GetInputDataType(1)) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, dtype);
}
}  // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}  // namespace ops
