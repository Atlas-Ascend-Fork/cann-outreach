// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    // 启动核数：实测 4 最快（8 慢、2 也慢）
    constexpr uint32_t BLOCK_DIM = 4;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        uint32_t elemBytes = ge::GetSizeByDataType(dtype_x);
        uint32_t alignElements = 32u / elemBytes;
        uint32_t length_x = tensor_x->GetShapeSize();
        uint32_t totalBlocks = length_x / alignElements;

        uint32_t blockDim = BLOCK_DIM;
        if (blockDim > totalBlocks) {
            blockDim = totalBlocks;
        }
        if (blockDim == 0) {
            blockDim = 1;
        }

        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        tiling->blocksPerCore = totalBlocks / blockDim;
        tiling->extra = totalBlocks % blockDim;

        context->SetBlockDim(blockDim);

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