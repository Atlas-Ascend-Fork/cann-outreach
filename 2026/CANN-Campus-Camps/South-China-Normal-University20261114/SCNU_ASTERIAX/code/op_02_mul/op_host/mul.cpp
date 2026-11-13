#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <limits>
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *x = context->GetRequiredInputTensor(0);
    const auto *y = context->GetRequiredInputTensor(1);
    const auto &shape = context->GetInputShape(0)->GetStorageShape();
    const auto &yShape = context->GetInputShape(1)->GetStorageShape();
    // Any concrete ND shape is supported; inputs must match, without broadcasting.
    if (yShape != shape || x->GetDataType() != y->GetDataType() ||
        (x->GetDataType() != ge::DT_FLOAT && x->GetDataType() != ge::DT_FLOAT16)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t totalLength = 1; // A rank-zero tensor contains one scalar.
    bool empty = false;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return ge::GRAPH_FAILED; // Runtime tiling requires concrete dimensions.
        }
        empty = empty || dim == 0;
    }
    if (empty) {
        totalLength = 0;
    } else {
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            const uint64_t dim = static_cast<uint64_t>(shape.GetDim(i));
            if (totalLength > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / dim) {
                return ge::GRAPH_FAILED;
            }
            totalLength *= dim;
        }
    }
    const uint32_t dtype = static_cast<uint32_t>(x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtype);
    auto *tiling = context->GetTilingData<MulTilingData>();
    tiling->totalLength = totalLength;
    tiling->tileLength = 2048;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tiles = totalLength / tiling->tileLength +
        (totalLength % tiling->tileLength != 0);
    uint32_t blockNum = cores < 8 ? cores : 8;
    if (tiles < blockNum) {
        blockNum = static_cast<uint32_t>(tiles);
    }
    context->SetBlockDim(blockNum == 0 ? 1 : blockNum);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name)
    {
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
        this->AICore().SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(Mul);
} // namespace ops
