// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const auto *shapeX = context->GetInputShape(0);
        const auto *shapeY = context->GetInputShape(1);
        const auto *descX = context->GetInputDesc(0);
        const auto *descY = context->GetInputDesc(1);
        if (shapeX == nullptr || shapeY == nullptr || descX == nullptr || descY == nullptr) {
            return ge::GRAPH_FAILED;
        }

        constexpr int64_t INPUT_LENGTH = 8 * 2048;
        const auto dtype = descX->GetDataType();
        const int64_t length = shapeX->GetStorageShape().GetShapeSize();
        if (length != INPUT_LENGTH || shapeY->GetStorageShape().GetShapeSize() != length ||
            descY->GetDataType() != dtype || (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)) {
            return ge::GRAPH_FAILED;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const int32_t availableCores = platform.GetCoreNumAiv();
        if (availableCores <= 0) {
            return ge::GRAPH_FAILED;
        }
        // 核数充足时，每个输入在每核分配 4 KiB。
        const int32_t targetCores = dtype == ge::DT_FLOAT ? 16 : 8;
        const uint32_t blockDim = availableCores < targetCores ? availableCores : targetCores;
        const uint32_t groups = static_cast<uint32_t>(length) / MUL_GROUP_ELEMENTS;

        auto *tiling = context->GetTilingData<MulTilingData>();
        auto *workspace = context->GetWorkspaceSizes(1);
        if (tiling == nullptr || workspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->baseLength = (groups / blockDim) * MUL_GROUP_ELEMENTS;
        tiling->extraCores = groups % blockDim;

        const uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        context->SetBlockDim(blockDim);
        workspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static bool IsMulInputShape(const gert::Shape *shape) {
        return shape != nullptr && shape->GetDimNum() == 2 &&
            shape->GetDim(0) == 8 && shape->GetDim(1) == 2048;
    }

    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto *x = context->GetInputShape(0);
        const auto *y = context->GetInputShape(1);
        auto *z = context->GetOutputShape(0);
        if (!IsMulInputShape(x) || !IsMulInputShape(y) || z == nullptr) {
            return GRAPH_FAILED;
        }
        *z = *x;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto dtype = context->GetInputDataType(0);
        if (context->GetInputDataType(1) != dtype || (dtype != DT_FLOAT && dtype != DT_FLOAT16)) {
            return GRAPH_FAILED;
        }
        return context->SetOutputDataType(0, dtype);
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
