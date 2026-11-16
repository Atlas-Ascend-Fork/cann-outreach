// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 1. 输入 dtype 决定走哪个模板实例（0=DT_FLOAT, 1=DT_FLOAT16）
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
    uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 2. 取 tiling 结构并赋值
    MulTilingData *tiling = context->GetTilingData<MulTilingData>();

    uint32_t totalLength = static_cast<uint32_t>(tensor_x->GetShapeSize());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    // 数据量太小就少开核，保证每核至少能分到 TILE_NUM * BUFFER_NUM 个元素
    uint32_t blockDim = BLOCK_DIM;
    if (coreNum != 0 && coreNum < blockDim) {
        blockDim = coreNum;
    }
    while (blockDim > 1 && totalLength / blockDim < TILE_NUM * 2) {
        blockDim >>= 1;
    }
    context->SetBlockDim(blockDim);

    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    // 3. workspace（仅用基础 API，不需要额外空间）
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    // z 的 shape 与 x 完全一致
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    // z 的 dtype 与 x 一致
    ge::DataType dtype = context->GetInputDataType(0);
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
