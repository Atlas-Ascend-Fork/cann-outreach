// Host: validate the task inputs, select dtype and generate safe aligned tiling.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *x = context->GetRequiredInputTensor(0);
    const gert::Tensor *y = context->GetRequiredInputTensor(1);
    if (x == nullptr || y == nullptr) return ge::GRAPH_FAILED;
    const ge::DataType dtype = x->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        y->GetDataType() != dtype || x->GetShapeSize() != 16384 ||
        y->GetShapeSize() != x->GetShapeSize()) return ge::GRAPH_FAILED;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores <= 0) return ge::GRAPH_FAILED;
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t elementBytes = ge::GetSizeByDataType(dtype);
    const uint32_t length = static_cast<uint32_t>(x->GetShapeSize());
    uint32_t blockDim = 8;
    while (blockDim > static_cast<uint32_t>(availableCores)) blockDim /= 2;
    const uint32_t blockLength = length / blockDim;
    // Baseline: 128 elements per tile, 16 iterations per core on eight cores.
    uint32_t tileLength = 128;
    const uint64_t reservedBytes = 4096;
    if (ubSize < reservedBytes + 3ULL * tileLength * elementBytes)
        return ge::GRAPH_FAILED;

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    if (tiling == nullptr) return ge::GRAPH_FAILED;
    tiling->length = length;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    tiling->tileNum = blockLength / tileLength;
    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) return ge::GRAPH_FAILED;
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x = context->GetInputShape(0);
    const gert::Shape *y = context->GetInputShape(1);
    gert::Shape *z = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr || z == nullptr ||
        x->GetDimNum() != 2 || y->GetDimNum() != 2 ||
        x->GetDim(0) != 8 || x->GetDim(1) != 2048 ||
        y->GetDim(0) != 8 || y->GetDim(1) != 2048) return GRAPH_FAILED;
    *z = *x;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto dtype = context->GetInputDataType(0);
    if ((dtype != DT_FLOAT && dtype != DT_FLOAT16) ||
        context->GetInputDataType(1) != dtype) return GRAPH_FAILED;
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}
