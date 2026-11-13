// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        const auto* input_shape_x = context->GetInputShape(0);
        const auto* input_shape_y = context->GetInputShape(1);
        const auto* desc_x = context->GetInputDesc(0);
        const auto* desc_y = context->GetInputDesc(1);

        if (input_shape_x == nullptr || input_shape_y == nullptr ||
            desc_x == nullptr || desc_y == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const auto& shape_x = input_shape_x->GetStorageShape();
        const auto& shape_y = input_shape_y->GetStorageShape();

        if (shape_x.GetDimNum() != 2 || shape_y.GetDimNum() != 2 ||
            shape_x.GetDim(0) != 8 || shape_x.GetDim(1) != 2048 ||
            shape_y.GetDim(0) != 8 || shape_y.GetDim(1) != 2048) {
            return ge::GRAPH_FAILED;
        }

        const ge::DataType dtype_x = desc_x->GetDataType();

        if (dtype_x != desc_y->GetDataType() ||
            (dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16)) {
            return ge::GRAPH_FAILED;
        }

        auto platform = platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());
        const uint32_t available_cores = platform.GetCoreNumAiv();

        if (available_cores == 0) {
            return ge::GRAPH_FAILED;
        }

        const uint32_t length =
            static_cast<uint32_t>(shape_x.GetShapeSize());
        const uint32_t tile_count = length / MUL_TILE_ELEMENTS;
        const uint32_t block_dim =
            available_cores < tile_count ? available_cores : tile_count;

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData* tiling =
            context->GetTilingData<MulTilingData>();

        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }

        tiling->length = length;
        context->GetRawTilingData()->SetDataSize(sizeof(MulTilingData));
        context->SetBlockDim(block_dim);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);

        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }

        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const auto* input_shape = context->GetInputShape(0);
        auto* output_shape = context->GetOutputShape(0);

        if (input_shape == nullptr || output_shape == nullptr) {
            return GRAPH_FAILED;
        }

        *output_shape = *input_shape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto dtype_x = context->GetInputDataType(0);

        if (dtype_x != context->GetInputDataType(1) ||
            (dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16)) {
            return ge::GRAPH_FAILED;
        }

        return context->SetOutputDataType(0, dtype_x);
    }
}  // namespace ge

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char *name) : OpDef(name)
        {
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
