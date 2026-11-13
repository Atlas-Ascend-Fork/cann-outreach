// Host side: tiling + shape/dtype inference
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // input tensor info
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t dt_x = static_cast<uint32_t>(dtype_x);
        // tiling key: tell the kernel which dtype it runs with
        ASCENDC_TPL_SEL_PARAM(context, dt_x);
        // fill the tiling data
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        const gert::StorageShape *x_shape = context->GetInputShape(0);
        int64_t total = 1;
        for (size_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
            total *= x_shape->GetStorageShape().GetDim(i);
        }
        tiling->length = static_cast<uint32_t>(total);
        // block dim
        context->SetBlockDim(8);
        // workspace size
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // elementwise op: output shape == input shape
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // elementwise op: output dtype == input dtype
        auto input_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_dtype);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char *name) : OpDef(name) {
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
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Mul);
}  // namespace ops