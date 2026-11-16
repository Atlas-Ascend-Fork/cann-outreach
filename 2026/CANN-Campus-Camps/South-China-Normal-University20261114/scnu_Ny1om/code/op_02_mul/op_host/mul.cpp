// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t BLOCK_SIZE = 32; // 32B对齐

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        if (num_cores_aiv == 0) num_cores_aiv = 8; // 兜底

        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        uint32_t length_x = tensor_x->GetShapeSize(); 
        uint32_t dtype_size = ge::GetSizeByDataType(tensor_x->GetDataType());
        
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        
        // 计算 Tiling 策略：多核切分
        // 向上对齐到 32B，保证每次搬运的数据量满足对齐要求
        uint32_t totalLengthAligned = (length_x * dtype_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / dtype_size;
        
        uint32_t blockLength = totalLengthAligned / num_cores_aiv;
        blockLength = (blockLength * dtype_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / dtype_size;
        
        uint32_t tileNum = 8; // 每个核内部切分 8 块
        uint32_t tileLength = blockLength / tileNum;
        tileLength = (tileLength * dtype_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / dtype_size;

        tiling->totalLength = length_x;
        tiling->blockLength = blockLength;
        tiling->tileNum = tileNum;
        tiling->tileLength = tileLength;

        context->SetBlockDim(num_cores_aiv);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
