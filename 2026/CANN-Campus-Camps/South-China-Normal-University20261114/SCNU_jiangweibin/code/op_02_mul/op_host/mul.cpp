// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t availableCores = platform.GetCoreNumAiv();
        const auto* inputShape = context->GetInputShape(0);
        const auto* inputDesc = context->GetInputDesc(0);
        if (availableCores == 0 || inputShape == nullptr || inputDesc == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const int64_t length = inputShape->GetStorageShape().GetShapeSize();
        if (length != 8 * 2048) {
            return ge::GRAPH_FAILED;
        }

        constexpr uint32_t tileLength = 256;
        const uint32_t tiles = length / tileLength;
        uint32_t cores = availableCores < tiles ? availableCores : tiles;
        while (tiles % cores != 0) {
            --cores;
        }

        uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        auto* tiling = context->GetTilingData<MulTilingData>();
        tiling->length = static_cast<uint32_t>(length);
        context->SetBlockDim(cores);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto* input = context->GetInputShape(0);
        auto* output = context->GetOutputShape(0);
        if (input == nullptr || output == nullptr) {
            return GRAPH_FAILED;
        }
        *output = *input;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
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
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Mul);
}  // namespace ops
