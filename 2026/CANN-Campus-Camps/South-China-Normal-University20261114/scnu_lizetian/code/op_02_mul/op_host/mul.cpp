#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const gert::Tensor* tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor* tensor_y =
        context->GetRequiredInputTensor(1);

    ge::DataType dtype_x = tensor_x->GetDataType();
    ge::DataType dtype_y = tensor_y->GetDataType();

    // 两个输入必须使用相同的数据类型
    if (dtype_x != dtype_y) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入元素总数
    uint32_t length_x = tensor_x->GetShapeSize();

    // 当前题目固定为 8 * 2048
    if (length_x != 8 * 2048) {
        return ge::GRAPH_FAILED;
    }

    // 配置数据类型模板参数
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 填充Tiling数据
    MulTilingData* tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length = length_x;

    // 使用8个AIV核，每个核处理2048个元素
    context->SetBlockDim(4);

    // Workspace不需要额外空间
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    *context->GetOutputShape(0) =
        *context->GetInputShape(0);

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(
        0,
        context->GetInputDataType(0)
    );

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