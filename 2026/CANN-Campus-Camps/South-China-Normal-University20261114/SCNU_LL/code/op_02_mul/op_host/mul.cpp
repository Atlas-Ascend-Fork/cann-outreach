#include <register/op_def_registry.h>
#include <tiling/platform/platform_ascendc.h>
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    
    MulTilingData *tiling = context->GetTilingData<MulTilingData>();

    // 获取输入 Shape 计算总元素个数
    const auto xShapePtr = context->GetInputShape(0);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    uint64_t totalLength = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        totalLength *= static_cast<uint64_t>(xShape.GetDim(i));
    }

    // 填充 tiling
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->tileNum = TILE_NUM;

    // 【修复 361001 关键 1】配置 BlockDim (启动的核数)，必须大于 0
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) coreNum = 8; // 降级保底方案
    context->SetBlockDim(coreNum);

    // 【修复 361001 关键 2】配置 Workspace 大小，不能为 0，给个 16MB
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 16 * 1024 * 1024;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    // 将输入的 Shape 赋值给输出的 Shape
    auto input_shape = context->GetInputShape(0);
    auto output_shape = context->GetOutputShape(0);
    *output_shape = *input_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    // 将输入的数据类型赋值给输出
    auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
} // namespace ops
