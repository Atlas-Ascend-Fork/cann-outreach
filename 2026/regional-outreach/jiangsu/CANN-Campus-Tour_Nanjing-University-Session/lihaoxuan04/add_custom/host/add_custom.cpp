/*
 * add_custom 算子 host 侧实现
 * - tiling 计算：根据输入 shape 计算 totalLength / tileNum，按 8 个 AI Core 切分
 * - 算子注册：输入 x、y，输出 z，均为 float16 / ND
 */
#include <cmath>

#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM = 8;    // 使用的 AI Core 数
constexpr uint32_t TILE_LENGTH = 256; // 单次搬运/计算的理想长度（元素数）

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AddCustomTilingData tiling;

    // 输入 x、y 同 shape，2 维 [N2, N1]，按元素总数拉平处理
    const gert::StorageShape *xShape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (auto i : xShape->GetStorageShape()) {
        totalLength *= static_cast<uint32_t>(i);
    }

    // 约束：totalLength 需能被 BLOCK_DIM * BUFFER_NUM(2) * TILE_LENGTH 整除
    // 例如 [64, 64] -> 4096 = 8 * 2 * 256 * 1，满足
    const uint32_t unit = BLOCK_DIM * 2 * TILE_LENGTH;
    if (totalLength == 0 || totalLength % unit != 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(BLOCK_DIM);

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(totalLength / unit);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    *zShape = *xShape; // z 与 x 同 shape
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
    }
};

OP_ADD(AddCustom);

} // namespace ops
