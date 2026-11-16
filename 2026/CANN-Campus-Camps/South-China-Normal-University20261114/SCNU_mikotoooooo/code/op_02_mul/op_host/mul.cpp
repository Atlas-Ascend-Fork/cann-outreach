
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {

        // 获取输入张量
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);

        // 获取输入数据类型
        ge::DataType dtype_x = tensor_x->GetDataType();

        // 配置 Tiling Key
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 获取 Tiling 数据结构
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();

        // 获取输入元素总数
        uint32_t length_x = tensor_x->GetShapeSize();

        // 设置 Tiling 参数
        tiling->length = length_x;
        tiling->tileNum = 8;

        // 固定使用 8 个 Block，与 Kernel 分块策略保持一致
        context->SetBlockDim(8);

        // 本算子不需要额外 Workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }

}  // namespace optiling


namespace ge {

    // 推导输出张量形状
    static graphStatus InferShape(gert::InferShapeContext *context) {

        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* z_shape = context->GetOutputShape(0);

        *z_shape = *x_shape;

        return GRAPH_SUCCESS;
    }

    // 推导输出数据类型
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {

        const auto inputDataType = context->GetInputDataType(0);

        context->SetOutputDataType(0, inputDataType);

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

            this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };

    OP_ADD(Mul);

}  // namespace ops
