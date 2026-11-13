
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    const gert::Tensor* tensor_x =
        context->GetRequiredInputTensor(0);

    if (tensor_x == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype_x =
        tensor_x->GetDataType();

    uint32_t length_x =
        tensor_x->GetShapeSize();

    // 保持模板原有TilingKey配置
    uint32_t DT_X =
        static_cast<uint32_t>(dtype_x);

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData* tiling =
        context->GetTilingData<MulTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->length = length_x;

    
    constexpr uint32_t BLOCK_DIM = 4;
    context->SetBlockDim(BLOCK_DIM);

    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const gert::Shape* inputShape =
        context->GetInputShape(0);

    gert::Shape* outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;

    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const auto inputDataType =
        context->GetInputDataType(0);

    return context->SetOutputDataType(
        0, inputDataType);
}

} // namespace ge

namespace ops {

class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

} // namespace ops
