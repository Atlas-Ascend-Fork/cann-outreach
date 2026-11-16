// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        
        // === TODO 1: 考生自行补齐 (Tiling 参数计算) ===
        // 1. 获取输入 x 的 Shape 并计算总元素个数
        auto x_shape = context->GetInputShape(0)->GetStorageShape();
        uint32_t totalLength = 1;
        for (size_t i = 0; i < x_shape.GetDimNum(); ++i) {
            totalLength *= x_shape.GetDim(i);
        }
        
        // 2. 填充 TilingData 结构体
        tiling->totalLength = totalLength;
        tiling->tileNum = TILE_NUM;
        
        // 3. 设置核数 (根据题目通常设为 8，你也可以通过 platform 动态获取)
        context->SetBlockDim(8); 
        // ================================================

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // === TODO 2: 考生自行补齐 (形状推导) ===
        // 矢量乘法是逐元素操作，输出 z 的 shape 与输入 x 的 shape 完全一致
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        // =======================================
        return GRAPH_SUCCESS;
    }
    
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // === TODO 3: 考生自行补齐 (数据类型推导) ===
        // 输出 z 的数据类型与输入 x 的数据类型完全一致
        auto dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtype);
        // ==========================================
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