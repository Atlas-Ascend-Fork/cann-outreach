// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 1. 按输入 dtype 选择模板实例（fp32 / fp16）
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 2. 计算数据总长度（8 * 2048 = 16384），填充 tiling 结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        int32_t data_sz = 1;
        for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
            data_sz *= x1_shape->GetStorageShape().GetDim(i);
        }
        tiling->length = data_sz;      // 数据总长度（字段名与 mul_tiling.h 一致）
        tiling->tileNum = TILE_NUM;    // 单核内分块数

        // 3. 配置启动核数：必须为 8，否则 16384 不能整除会导致尾块丢失
        context->SetBlockDim(8);

        // 4. 配置 workspace 大小（本算子不需要额外 workspace）
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 逐元素算子：输出 z 的 shape 与输入 x 完全一致
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* z_shape = context->GetOutputShape(0);
        *z_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出 dtype 与输入 x 一致
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char* name) : OpDef(name) {
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