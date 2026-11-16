// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const auto* shape_x = context->GetInputShape(0);
        const auto* shape_y = context->GetInputShape(1);
        const auto* desc_x = context->GetInputDesc(0);
        const auto* desc_y = context->GetInputDesc(1);
        if (shape_x == nullptr || shape_y == nullptr ||
            desc_x == nullptr || desc_y == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto& x_shape = shape_x->GetStorageShape();
        const auto& y_shape = shape_y->GetStorageShape();
        if (x_shape.GetDimNum() != y_shape.GetDimNum()) {
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0; i < x_shape.GetDimNum(); ++i) {
            if (x_shape.GetDim(i) < 0 || x_shape.GetDim(i) != y_shape.GetDim(i)) {
                return ge::GRAPH_FAILED;
            }
        }
        const ge::DataType dtype_x = desc_x->GetDataType();
        if ((dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16) ||
            dtype_x != desc_y->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
        const int64_t length = x_shape.GetShapeSize();
        if (length < 0 || static_cast<uint64_t>(length) > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t length_x = static_cast<uint32_t>(length);
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t core_count = platform.GetCoreNumAiv();
        if (core_count == 0) {
            return ge::GRAPH_FAILED;
        }
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->length = length_x;
        const uint32_t tile_count = length_x / MUL_TILE_LENGTH +
                                    (length_x % MUL_TILE_LENGTH != 0);
        const uint32_t active_cores = tile_count < core_count ? tile_count : core_count;
        context->SetBlockDim(active_cores == 0 ? 1 : active_cores);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto* x_shape = context->GetInputShape(0);
        auto* z_shape = context->GetOutputShape(0);
        if (x_shape == nullptr || z_shape == nullptr) {
            return GRAPH_FAILED;
        }
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto dtype = context->GetInputDataType(0);
        if ((dtype != DT_FLOAT && dtype != DT_FLOAT16) ||
            dtype != context->GetInputDataType(1)) {
            return GRAPH_FAILED;
        }
        return context->SetOutputDataType(0, dtype);
    }
}  

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
}  
