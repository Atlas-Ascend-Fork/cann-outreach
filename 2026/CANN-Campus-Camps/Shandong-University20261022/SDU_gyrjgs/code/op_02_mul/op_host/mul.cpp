
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    // 获取输入张量
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);

    // 获取输入数据类型
    ge::DataType dtype_x = tensor_x->GetDataType();

    // 获取输入元素总数
    uint32_t length_x =
        static_cast<uint32_t>(tensor_x->GetShapeSize());

    // 配置数据类型模板
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 配置 tiling 参数
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length = length_x;
    tiling->tileLength = 256;

    // 每个核处理一个或多个数据块
    uint32_t tile_num =
        (length_x + tiling->tileLength - 1) /
        tiling->tileLength;

    uint32_t block_dim =
        (num_cores_aiv < static_cast<int32_t>(tile_num))
        ? static_cast<uint32_t>(num_cores_aiv)
        : tile_num;

    context->SetBlockDim(block_dim);

    // 本算子不需要额外 workspace
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context) {

    const gert::Shape *input_shape =
        context->GetInputShape(0);

    gert::Shape *output_shape =
        context->GetOutputShape(0);

    *output_shape = *input_shape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context) {

    context->SetOutputDataType(
        0, context->GetInputDataType(0));

    return GRAPH_SUCCESS;
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops
