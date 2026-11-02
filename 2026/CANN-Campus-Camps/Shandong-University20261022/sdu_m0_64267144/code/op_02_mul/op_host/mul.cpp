#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    const int32_t coreCount = platform.GetCoreNumAiv();

    const gert::Tensor* x = context->GetRequiredInputTensor(0);
    const gert::Tensor* y = context->GetRequiredInputTensor(1);
    if (x == nullptr || y == nullptr || coreCount <= 0) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = x->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        y->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }

    const auto elementCount = x->GetShapeSize();
    if (elementCount <= 0 || elementCount > UINT32_MAX ||
        y->GetShapeSize() != elementCount) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t length = static_cast<uint32_t>(elementCount);
    constexpr uint32_t tileLength = 256;

    // The specified (8, 2048) input has 16384 elements, or 64 full tiles.
    // Both float and half transfers are multiples of 32 bytes.
    if (length % tileLength != 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tileCount = length / tileLength;

    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData* tiling = context->GetTilingData<MulTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = length;
    tiling->tileLength = tileLength;
    tiling->tileCount = tileCount;

    const uint32_t available = static_cast<uint32_t>(coreCount);
    const uint32_t blockDim = available < tileCount ? available : tileCount;
    context->SetBlockDim(blockDim);

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* zShape = context->GetOutputShape(0);
    if (xShape == nullptr || zShape == nullptr) {
        return GRAPH_FAILED;
    }
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char* name) : OpDef(name)
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
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}  // namespace ops
