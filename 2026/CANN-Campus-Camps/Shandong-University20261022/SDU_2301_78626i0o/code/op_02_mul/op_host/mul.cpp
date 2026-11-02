// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

// 每块最多处理2048个元素，兼容float32和float16的32字节对齐。
constexpr uint32_t TILE_LENGTH = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 获取平台可用核数。
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores <= 0) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入数据类型。
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    ge::DataType dtype_x = tensor_x->GetDataType();

    if (dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    // 根据输入shape计算总元素数。
    const gert::StorageShape *x_shape = context->GetInputShape(0);
    const auto &shape = x_shape->GetStorageShape();

    uint64_t totalLength = 1;
    constexpr uint64_t MAX_LENGTH = 0xFFFFFFFFULL;

    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t dim = shape.GetDim(i);

        if (dim < 0) {
            return ge::GRAPH_FAILED;
        }

        uint64_t dimLength = static_cast<uint64_t>(dim);
        if (dimLength != 0 && totalLength > MAX_LENGTH / dimLength) {
            return ge::GRAPH_FAILED;
        }

        totalLength *= dimLength;
    }

    // 配置数据类型模板参数。
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 根据数据块数量选择启动核数。
    uint64_t totalTiles =
        (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;

    uint32_t blockDim = 1;
    if (totalTiles > 0) {
        blockDim =
            totalTiles < static_cast<uint64_t>(availableCores)
                ? static_cast<uint32_t>(totalTiles)
                : static_cast<uint32_t>(availableCores);
    }

    // 每个核分配整数个tile，保证各核起始地址对齐。
    uint64_t tilesPerCore =
        (totalTiles + blockDim - 1) / blockDim;

    uint64_t blockLength = tilesPerCore * TILE_LENGTH;
    if (blockLength > MAX_LENGTH) {
        return ge::GRAPH_FAILED;
    }

    // 填充Host和Kernel共用的tiling结构体。
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->blockLength = static_cast<uint32_t>(blockLength);
    tiling->tileLength = TILE_LENGTH;

    context->SetBlockDim(blockDim);

    // 不使用额外workspace。
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    // 输出z的shape与输入x一致。
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *z_shape = context->GetOutputShape(0);

    *z_shape = *x_shape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 输出z的数据类型与输入x一致。
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);

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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops