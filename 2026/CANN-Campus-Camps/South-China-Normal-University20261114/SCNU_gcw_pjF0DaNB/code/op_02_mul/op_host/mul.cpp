// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0; 
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
        }

        auto xShape = context->GetInputShape(0)->GetStorageShape();
        uint32_t totalLength = 1;
        for (size_t i = 0; i < xShape.GetDimNum(); i++) {
            totalLength *= xShape.GetDim(i);
        }
        tiling->length = totalLength;

        // 根据 dtype 取模板参数值，传给 ASCENDC_TPL_SEL_PARAM
         uint32_t dtypeX = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, dtypeX);

        context->SetBlockDim(8);
        return ge::GRAPH_SUCCESS;
    }

} // namespace optiling


namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* zShape = context->GetOutputShape(0);
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}

} // namespace ge


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
} // namespace ops