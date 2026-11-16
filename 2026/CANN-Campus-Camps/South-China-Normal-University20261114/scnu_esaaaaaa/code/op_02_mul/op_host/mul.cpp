// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        uint64_t ub_size;
        platform.GetCoreMemSize(
            platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor* tensor_x =
            context->GetRequiredInputTensor(0);
        const gert::Tensor* tensor_y =
            context->GetRequiredInputTensor(1);

        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();
        uint32_t size_x = tensor_x->GetSize();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 选择可整除输入总长度的 2 的幂次核数，
        // 确保每个核处理同样大小且对齐的数据分片。
        uint32_t block_dim = 1;
        const uint32_t core_limit = num_cores_aiv > 0
            ? static_cast<uint32_t>(num_cores_aiv)
            : 1;

        while (block_dim <= core_limit / 2 &&
               block_dim <= length_x / 32 &&
               length_x % (block_dim * 2) == 0) {
            block_dim *= 2;
        }

        MulTilingData* tiling =
            context->GetTilingData<MulTilingData>();
        tiling->length = length_x;
        tiling->blockLength = length_x / block_dim;

        context->SetBlockDim(block_dim);

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