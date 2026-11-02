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
        tensorY->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }
    const int64_t length = tensorX->GetShapeSize();
    if (length < 0 || length > std::numeric_limits<uint32_t>::max() ||
        tensorY->GetShapeSize() != length) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t availableCores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

    MulTilingData* tiling = context->GetTilingData<MulTilingData>();
    if (tiling == nullptr || availableCores <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalLength = static_cast<uint32_t>(length);
    const uint32_t elementBytes = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    const uint32_t alignElems = 512u / elementBytes;

    // Tunable starting point for the specified 16384-element workload.
    constexpr uint32_t MAX_CORES = 8u;
    constexpr uint64_t UB_RESERVE_BYTES = 4096u;
    constexpr uint64_t MAX_TILE_BYTES = 64u * 1024u;

    uint32_t blockDim = 1;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    if (totalLength != 0) {
        if (ubBytes <= UB_RESERVE_BYTES) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t units = (static_cast<uint64_t>(totalLength) + alignElems - 1u) / alignElems;
        blockDim = std::min<uint32_t>(MAX_CORES, static_cast<uint32_t>(availableCores));
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(blockDim, units));

        const uint64_t unitsPerCore = (units + blockDim - 1u) / blockDim;
        const uint64_t blockLength64 = unitsPerCore * alignElems;
        if (blockLength64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        blockLength = static_cast<uint32_t>(blockLength64);
        // Remove cores whose aligned data slice would be empty.
        blockDim = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalLength) + blockLength - 1u) / blockLength);

        // Two input regions; the second is also the output region.
        const uint64_t maxInputBytes = std::min<uint64_t>(
            (ubBytes - UB_RESERVE_BYTES) / 2u, MAX_TILE_BYTES);
        const uint32_t maxTileLength = static_cast<uint32_t>(maxInputBytes / 512u) * alignElems;
        if (maxTileLength == 0) {
            return ge::GRAPH_FAILED;
        }
        tileLength = std::min(blockLength, maxTileLength);
    }

    tiling->length = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    context->SetBlockDim(blockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
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
    context->SetOutputDataType(0, context->GetInputDataType(0));
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

} // namespace ops
