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
    const auto* shapeX = context->GetInputShape(0);
    const auto* shapeY = context->GetInputShape(1);
    const auto* descX = context->GetInputDesc(0);
    const auto* descY = context->GetInputDesc(1);

    if (shapeX == nullptr || shapeY == nullptr ||
        descX == nullptr || descY == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 只支持形状一致的逐元素乘法。
    const auto& xShape = shapeX->GetStorageShape();
    const auto& yShape = shapeY->GetStorageShape();

    if (xShape.GetDimNum() != yShape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        if (xShape.GetDim(i) != yShape.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }

    const ge::DataType dtype = descX->GetDataType();

    if (dtype != descY->GetDataType() ||
        (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t length = xShape.GetShapeSize();

    // 当前 Tiling 的 length 使用 uint32_t。
    if (length < 0 ||
        static_cast<uint64_t>(length) >
            std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const uint32_t elementBytes =
        static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    const uint32_t alignElements = 32U / elementBytes;

    // 三个队列各使用一个 buffer。
    // 每个 buffer 最多 16 KiB，三者合计不超过 UB 的一半。
    uint64_t tileBytes = std::min<uint64_t>(16U * 1024U, ubSize / 6U);
    tileBytes = tileBytes / 32U * 32U;

    if (tileBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<MulTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->length = static_cast<uint32_t>(length);
    tiling->tileLength =
        static_cast<uint32_t>(tileBytes / elementBytes);

    // 按 32 字节分块；空输入启动一个核，Kernel 直接返回。
    const uint64_t alignedBlocks =
        (static_cast<uint64_t>(length) + alignElements - 1U) /
        alignElements;

    uint32_t blockDim = 1;
    if (alignedBlocks != 0) {
        blockDim = static_cast<uint32_t>(
            std::min<uint64_t>(
                alignedBlocks, static_cast<uint64_t>(availableCores)));
    }

    context->SetBlockDim(blockDim);

    const uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* xShape = context->GetInputShape(0);
    const auto* yShape = context->GetInputShape(1);
    auto* zShape = context->GetOutputShape(0);

    if (xShape == nullptr || yShape == nullptr || zShape == nullptr) {
        return GRAPH_FAILED;
    }

    if (xShape->GetDimNum() != yShape->GetDimNum()) {
        return GRAPH_FAILED;
    }

    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        if (xShape->GetDim(i) != yShape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }

    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const DataType dtype = context->GetInputDataType(0);

    if (dtype != context->GetInputDataType(1) ||
        (dtype != DT_FLOAT && dtype != DT_FLOAT16)) {
        return GRAPH_FAILED;
    }

    return context->SetOutputDataType(0, dtype);
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