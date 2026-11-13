// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;
    const uint32_t BLOCK_SIZE = 32; // 32 Bytes 对齐

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        
        // TODO: 考生自行补齐 (开始)
        // 1. 获取输入 shape 并计算总元素数
        const gert::StorageShape* x_shape = context->GetInputShape(0);
        int64_t data_sz = 1;
        for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
            data_sz *= x_shape->GetStorageShape().GetDim(i);
        }

        // 2. 32B 对齐策略（由于是单核且切分为 TILE_NUM * BUFFER_NUM 块，需保证每块 32B 对齐）
        // fp16 对齐 16 个元素，fp32 对齐 8 个元素。为了单次搬运满足对齐，总长度需对齐到足够大
        uint32_t align_elements = (DT_X == ge::DT_FLOAT16) ? (BLOCK_SIZE / 2) : (BLOCK_SIZE / 4);
        // 对齐长度向上取整
        uint32_t alignedLength = (data_sz + align_elements - 1) / align_elements * align_elements;

        tiling->totalLength = alignedLength;
        tiling->tileNum = TILE_NUM;

        // 3. 配置启动核数：要求单核实现
        context->SetBlockDim(1);
        // TODO: 考生自行补齐 (结束)

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // TODO: 考生自行补齐 (开始)
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* z_shape = context->GetOutputShape(0);
        // Mul 算子为逐元素运算，输出 shape 与输入一致
        *z_shape = *x_shape;
        // TODO: 考生自行补齐 (结束)
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // TODO: 考生自行补齐 (开始)
        const auto inputDataType = context->GetInputDataType(0);
        // 输出数据类型与输入一致
        context->SetOutputDataType(0, inputDataType);
        // TODO: 考生自行补齐 (结束)
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