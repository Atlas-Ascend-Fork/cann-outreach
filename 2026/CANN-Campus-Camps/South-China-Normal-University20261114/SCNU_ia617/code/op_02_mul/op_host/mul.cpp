#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取输入类型，用于选择 float / half 模板实例
    const gert::Tensor* tensor_x =
        context->GetRequiredInputTensor(0);

    uint32_t DT_X =
        static_cast<uint32_t>(tensor_x->GetDataType());

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取 Tiling 数据的写入位置
    MulTilingData* tiling =
        context->GetTilingData<MulTilingData>();

    // 计算输入总元素数
    const gert::StorageShape* x_shape =
        context->GetInputShape(0);

    uint32_t totalLength = 1;
    for (size_t i = 0;
         i < x_shape->GetStorageShape().GetDimNum();
         ++i) {
        totalLength *= static_cast<uint32_t>(
            x_shape->GetStorageShape().GetDim(i));
    }

    // 填写传给 Kernel 的参数
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    // 与加法参考代码一致，启动 8 个核
    context->SetBlockDim(8);

    // 本算子不需要额外 workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    // 本题两个输入形状相同，输出形状与输入 x 一致
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* z_shape = context->GetOutputShape(0);

    *z_shape = *x_shape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    // 输出数据类型与输入 x 一致
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);

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
