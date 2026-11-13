// Fixed-shape elementwise multiplication: (8, 2048), float16 / float32.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
constexpr uint32_t TOTAL_LENGTH = 8U * 2048U;
// Start with 8 cores. Change only this constant to 4 for an NPU comparison.
constexpr uint32_t PREFERRED_CORES = 4;
static_assert(PREFERRED_CORES == 1 || PREFERRED_CORES == 2 ||
              PREFERRED_CORES == 4 || PREFERRED_CORES == 8,
              "PREFERRED_CORES must be 1, 2, 4, or 8");

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *x = context->GetRequiredInputTensor(0);
    const gert::Tensor *y = context->GetRequiredInputTensor(1);
    if (x == nullptr || y == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = x->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        y->GetDataType() != dtype ||
        x->GetShapeSize() != TOTAL_LENGTH ||
        y->GetShapeSize() != TOTAL_LENGTH) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = PREFERRED_CORES;
    while (blockDim > static_cast<uint32_t>(availableCores)) {
        blockDim >>= 1;
    }
    const uint32_t blockLength = TOTAL_LENGTH / blockDim;

    // Check the actual platform UB before allocating three per-core buffers.
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t elementBytes = (dtype == ge::DT_FLOAT) ? 4U : 2U;
    if (3ULL * blockLength * elementBytes > ubSize) {
        return ge::GRAPH_FAILED;
    }

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (tiling == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockLength = blockLength;

    const uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    context->SetBlockDim(blockDim);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    if (xShape == nullptr || zShape == nullptr) {
        return GRAPH_FAILED;
    }
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}  // namespace ops
