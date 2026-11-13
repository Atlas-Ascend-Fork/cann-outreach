// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        const auto* xShape = context->GetInputShape(0);
        const auto* yShape = context->GetInputShape(1);
        const auto* xDesc = context->GetInputDesc(0);
        const auto* yDesc = context->GetInputDesc(1);

        if (!xShape || !yShape || !xDesc || !yDesc ||
            !context->GetPlatformInfo()) {
            return ge::GRAPH_FAILED;
        }

        const auto& xs = xShape->GetStorageShape();
        const auto& ys = yShape->GetStorageShape();

        if (xs.GetDimNum() != 2 || ys.GetDimNum() != 2 ||
            xs.GetDim(0) != 8 || xs.GetDim(1) != 2048 ||
            ys.GetDim(0) != 8 || ys.GetDim(1) != 2048 ||
            xDesc->GetDataType() != yDesc->GetDataType()) {
            return ge::GRAPH_FAILED;
        }

        const auto dtype = xDesc->GetDataType();
        if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
            return ge::GRAPH_FAILED;
        }

        uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        if (!tiling) {
            return ge::GRAPH_FAILED;
        }

        auto platform =
            platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t available = platform.GetCoreNumAiv();

        if (available == 0) {
            return ge::GRAPH_FAILED;
        }

        uint32_t blockDim = 8;
        while (blockDim > available) {
            blockDim /= 2;
        }

        tiling->totalLength = 8 * 2048;
        tiling->tileNum = TILE_NUM;
        context->SetBlockDim(blockDim);

        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext* context)
    {
        const auto* x = context->GetInputShape(0);
        const auto* y = context->GetInputShape(1);
        auto* z = context->GetOutputShape(0);

        if (!x || !y || !z || x->GetDimNum() != y->GetDimNum()) {
            return GRAPH_FAILED;
        }

        for (size_t i = 0; i < x->GetDimNum(); ++i) {
            if (x->GetDim(i) != y->GetDim(i)) {
                return GRAPH_FAILED;
            }
        }

        *z = *x;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext* context)
    {
        const auto dtype = context->GetInputDataType(0);

        if (dtype != context->GetInputDataType(1) ||
            (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)) {
            return ge::GRAPH_FAILED;
        }

        context->SetOutputDataType(0, dtype);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char* name) : OpDef(name)
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
}