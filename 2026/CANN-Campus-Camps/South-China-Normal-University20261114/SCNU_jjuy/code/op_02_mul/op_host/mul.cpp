#include "register/op_def_registry.h"

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
        y->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }

    // 本题输入固定为 (8, 2048)，共 16384 个元素。
    constexpr uint32_t TOTAL_LENGTH = 8 * 2048;
    constexpr uint32_t BLOCK_DIM = 8;
    if (x->GetShapeSize() != TOTAL_LENGTH ||
        y->GetShapeSize() != TOTAL_LENGTH) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = TOTAL_LENGTH;
    context->SetBlockDim(BLOCK_DIM);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    if (xShape == nullptr || zShape == nullptr) {
        return GRAPH_FAILED;
    }
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops
