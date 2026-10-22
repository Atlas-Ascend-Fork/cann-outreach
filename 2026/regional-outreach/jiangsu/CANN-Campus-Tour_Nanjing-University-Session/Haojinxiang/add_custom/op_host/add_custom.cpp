/**
 * @file add_custom.cpp
 * @brief add_custom 算子 Host 侧实现: TilingFunc + 算子原型 (OpDef) 注册。
 *
 * 算子: add_custom, z = x + y
 * shape: 2 维 [N2, N1]   dtype: float16   format: ND
 */
#include "register/op_def_registry.h"
#include "../op_kernel/add_custom_tiling.h"

namespace optiling {
const uint32_t NUM_BLOCKS = 8;  // 使用的 AI Core (Block) 数量
const uint32_t TILE_NUM = 8;    // 单核切分出的 tile 数量

// TilingFunc: 框架在算子执行前调用, 传入 gert::TilingContext*
// 前提: 总元素数 (N2*N1) 应能被 NUM_BLOCKS * TILE_NUM * 2 (双缓冲) 整除, 以保证切分精确
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AddCustomTilingData *tiling = context->GetTilingData<AddCustomTilingData>();
    // 读取输入 x 的原始 shape 并取总元素数 (2 维 => N2 * N1)
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    context->SetSimdNumBlocks(NUM_BLOCKS);  // 旧版 CANN 为 context->SetBlockDim(NUM_BLOCKS);
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;  // 本算子不使用 workspace
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
// 输出 shape 与输入 x 相同 (逐元素运算)
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *z_shape = context->GetOutputShape(0);
    *z_shape = *x_shape;
    return GRAPH_SUCCESS;
}

// 输出 dtype 与输入 x 相同 (float16)
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
    }
};
OP_ADD(AddCustom);
}  // namespace ops
