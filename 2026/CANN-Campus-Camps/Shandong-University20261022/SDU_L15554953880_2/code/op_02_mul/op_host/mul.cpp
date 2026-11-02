// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr || context->GetPlatformInfo() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        if (tensor_x == nullptr || tensor_y == nullptr || num_cores_aiv <= 0) {
            return ge::GRAPH_FAILED;
        }
        ge::DataType dtype_x = tensor_x->GetDataType();
        if ((dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16) ||
            tensor_y->GetDataType() != dtype_x) {
            return ge::GRAPH_FAILED;
        }
        int64_t elements = tensor_x->GetShapeSize();
        if (elements <= 0 || elements > 0xfffffff0LL ||
            tensor_y->GetShapeSize() != elements) {
            return ge::GRAPH_FAILED;
        }
        uint32_t length_x = static_cast<uint32_t>(elements);
        uint32_t dtype_size_x = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
        const uint32_t alignElements = 32 / dtype_size_x;

        // For this small vector, retain at least about 1024 elements per core.
        uint32_t blockNum = (length_x - 1) / 1024 + 1;
        if (blockNum > static_cast<uint32_t>(num_cores_aiv)) {
            blockNum = static_cast<uint32_t>(num_cores_aiv);
        }
        uint32_t coreElements = (length_x - 1) / blockNum + 1;
        uint32_t blockLength = (coreElements + alignElements - 1) / alignElements * alignElements;

        // Three double-buffered queues: x, y and z. Reserve UB for the runtime.
        const uint64_t reserveBytes = 1024;
        if (ub_size <= reserveBytes) return ge::GRAPH_FAILED;
        uint64_t tileCapacity = (ub_size - reserveBytes) / (6 * dtype_size_x);
        tileCapacity = tileCapacity / alignElements * alignElements;
        if (tileCapacity == 0) return ge::GRAPH_FAILED;
        uint32_t tileLength = blockLength < 4096 ? blockLength : 4096;
        if (tileCapacity < tileLength) tileLength = static_cast<uint32_t>(tileCapacity);

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        if (tiling == nullptr) return ge::GRAPH_FAILED;
        tiling->length = length_x;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;
        context->GetRawTilingData()->SetDataSize(sizeof(MulTilingData));
        context->SetBlockDim(blockNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) return ge::GRAPH_FAILED;
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) return GRAPH_FAILED;
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        if (inputShape == nullptr || outputShape == nullptr) return GRAPH_FAILED;
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) return ge::GRAPH_FAILED;
        const ge::DataType dtype = context->GetInputDataType(0);
        if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
            context->GetInputDataType(1) != dtype) {
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(0, dtype);
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
