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
        
        // 计算数据总长度
        uint32_t totalLength = tensor_x->GetOriginShape().GetShapeSize();
        uint32_t blockDim = TILE_NUM;
        
        // 防止数据量太小导致每块长度为0
        if (totalLength < blockDim) {
            blockDim = 1;
        }
        
        uint32_t tileLength = totalLength / blockDim;
        
        // 赋值给Tiling结构体
        tiling->totalLength = totalLength;
        tiling->blockDim = blockDim;
        tiling->tileLength = tileLength;
        
        // 设置BlockDim，给到硬件调度多核
        context->SetBlockDim(blockDim);
        
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 输出shape与输入x保持一致
        auto xShape = context->GetInputShape(0);
        auto zShape = context->GetOutputShape(0);
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }
    
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出dataType与输入x保持一致
        auto xDtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, xDtype);
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