// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;   // 每个核内的分块数
    const int32_t BLOCK_DIM = 8;   // 启动核数，需与 kernel 侧 GetBlockNum() 一致

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取输入张量信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        // 按数据类型选择 tiling key，实现 kernel 侧不同数据类型的区分
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 计算输入元素总个数（x 与 y 形状一致，以 x 为准）
        const gert::StorageShape *x_shape = context->GetInputShape(0);
        uint32_t totalLength = 1;
        for (int32_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
            totalLength *= static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(i));
        }

        // 填充 tiling 数据
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = totalLength;
        tiling->tileNum = TILE_NUM;

        // 配置启动核数
        context->SetBlockDim(BLOCK_DIM);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 逐元素算子：输出 z 的 shape 与输入 x 完全一致
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出数据类型与输入 x 保持一致
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
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Mul);
}  // namespace ops
