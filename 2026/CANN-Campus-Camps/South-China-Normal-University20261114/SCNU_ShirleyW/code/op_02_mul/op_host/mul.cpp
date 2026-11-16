#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);

    if (tensor_x == nullptr || tensor_y == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = tensor_x->GetDataType();
    if ((dtype != ge::DT_FLOAT &&
         dtype != ge::DT_FLOAT16) ||
        tensor_y->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }

    // 本题只支持形状 (8, 2048)。
    for (size_t i = 0; i < 2; ++i) {
        const auto *shape = context->GetInputShape(i);
        if (shape == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const auto &storage = shape->GetStorageShape();
        if (storage.GetDimNum() != 2 ||
            storage.GetDim(0) != 8 ||
            storage.GetDim(1) != 2048) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->length =
        static_cast<uint32_t>(tensor_x->GetShapeSize());

    context->SetBlockDim(BLOCK_DIM);

    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const auto *x = context->GetInputShape(0);
    const auto *y = context->GetInputShape(1);
    auto *z = context->GetOutputShape(0);

    if (x == nullptr || y == nullptr || z == nullptr ||
        x->GetDimNum() != y->GetDimNum()) {
        return GRAPH_FAILED;
    }

    for (size_t i = 0; i < x->GetDimNum(); ++i) {
        if (x->GetDim(i) >= 0 && y->GetDim(i) >= 0 &&
            x->GetDim(i) != y->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }

    *z = *x;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const auto dtype = context->GetInputDataType(0);

    if ((dtype != DT_FLOAT && dtype != DT_FLOAT16) ||
        context->GetInputDataType(1) != dtype) {
        return GRAPH_FAILED;
    }

    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}

} // namespace ge

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

} // namespace ops
