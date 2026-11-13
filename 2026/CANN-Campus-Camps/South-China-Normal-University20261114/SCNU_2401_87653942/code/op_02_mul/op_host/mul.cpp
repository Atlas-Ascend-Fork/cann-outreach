#include <algorithm>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const auto *tensorX = context->GetRequiredInputTensor(0);
        const auto *tensorY = context->GetRequiredInputTensor(1);
        const auto dtype = tensorX->GetDataType();
        if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
            tensorY->GetDataType() != dtype) {
            return ge::GRAPH_FAILED;
        }

        const auto &shape = context->GetInputShape(0)->GetStorageShape();
        const auto &shapeY = context->GetInputShape(1)->GetStorageShape();
        if (shape.GetDimNum() != shapeY.GetDimNum()) {
            return ge::GRAPH_FAILED;
        }
        for (size_t dim = 0; dim < shape.GetDimNum(); ++dim) {
            if (shape.GetDim(dim) != shapeY.GetDim(dim)) {
                return ge::GRAPH_FAILED;
            }
        }

        const int64_t shapeSize = shape.GetShapeSize();
        if (shapeSize < 0 || shapeSize > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }

        const auto length = static_cast<uint32_t>(shapeSize);
        const auto elementBytes = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
        const uint32_t elementsPerBlock = MUL_DATA_BLOCK_BYTES / elementBytes;

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t availableCores = platform.GetCoreNumAiv();
        uint64_t ubBytes = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

        // 两个输入队列和一个输出队列各使用一个 UB 缓冲区。
        constexpr uint64_t maxTileBytes = 16 * 1024;
        const uint64_t tileBytes = std::min(ubBytes / 3, maxTileBytes);
        const auto tileLength =
            static_cast<uint32_t>(tileBytes / MUL_DATA_BLOCK_BYTES) * elementsPerBlock;
        if (availableCores == 0 || tileLength == 0) {
            return ge::GRAPH_FAILED;
        }

        const uint32_t blockDim =
            std::min(availableCores, std::max(1U, length / elementsPerBlock));

        const uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        auto *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length;
        tiling->tileLength = tileLength;
        context->SetBlockDim(blockDim);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
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