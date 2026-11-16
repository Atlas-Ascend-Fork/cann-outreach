#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::Tensor* tensorX = context->GetRequiredInputTensor(0);
    const gert::Tensor* tensorY = context->GetRequiredInputTensor(1);
    if (tensorX == nullptr || tensorY == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = tensorX->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        tensorY->GetDataType() != dtype ||
        tensorX->GetStorageShape() != tensorY->GetStorageShape()) {
        return ge::GRAPH_FAILED;
    }

    const int64_t length = tensorX->GetShapeSize();
    const uint32_t dtypeSize = ge::GetSizeByDataType(dtype);
    const uint32_t alignElements = MUL_ALIGN_BYTES / dtypeSize;

    // 本实现使用对齐 DataCopy；
    // 题目中的 16384 个元素满足此条件。
    if (length <= 0 ||
        length > std::numeric_limits<uint32_t>::max() ||
        length % alignElements != 0) {
        return ge::GRAPH_FAILED;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const int32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ubSize);

    const uint64_t neededUb =
        3ULL * MUL_BUFFER_NUM * MUL_TILE_LENGTH * dtypeSize;

    if (ubSize < neededUb) {
        return ge::GRAPH_FAILED;
    }

    // 一个工作核至少分配一个 32 字节块。
    const uint32_t alignedBlocks =
        static_cast<uint32_t>(length) / alignElements;

    const uint32_t blockDim =
        std::min(static_cast<uint32_t>(coreNum), alignedBlocks);

    MulTilingData* tiling =
        context->GetTilingData<MulTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->length = static_cast<uint32_t>(length);
    context->SetBlockDim(blockDim);

    const uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* zShape = context->GetOutputShape(0);

    if (xShape == nullptr || zShape == nullptr) {
        return GRAPH_FAILED;
    }

    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto dtype = context->GetInputDataType(0);

    if ((dtype != DT_FLOAT && dtype != DT_FLOAT16) ||
        context->GetInputDataType(1) != dtype) {
        return GRAPH_FAILED;
    }

    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}

} // namespace ge

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

} // namespace ops