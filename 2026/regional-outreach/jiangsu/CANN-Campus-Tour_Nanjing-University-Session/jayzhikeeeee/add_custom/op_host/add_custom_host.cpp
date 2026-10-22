/**
 * add_custom 算子 —— 宿主侧（Host）代码
 * 包含：tiling 结构定义、tiling 计算函数、shape 推导、类型推导、算子注册
 *
 * 完整接入 CANN 框架（通过 Ascend C 算子工程模板）时，
 * 本文件与 op_host/add_custom_tiling.h 搭配使用。
 */
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

using namespace ge;
namespace ops {

/**
 * Tiling 计算函数（Host 侧执行）
 * 策略：
 *   - 每个 AI Core 处理 blockLength = totalLength / blockDim 个元素
 *   - 每次循环处理 tileLength 个元素（按 32B 对齐的元素数，即 16 个 half）
 */
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AddCustomTilingData tiling;
    const gert::StorageShape *xShape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    // ND 格式 2 维 [N2, N1]，展平计算总元素个数
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= xShape->GetStorageShape().GetDim(i);
    }

    // 单次 tile 处理长度：约 16KB 的 half 数据（8192 个元素），并按 16 对齐
    constexpr uint32_t ELEMENTS_PER_TILE = 8192;
    uint32_t tileLength = (totalLength >= ELEMENTS_PER_TILE)
                              ? ELEMENTS_PER_TILE
                              : ((totalLength + 15) / 16) * 16; // 32B 对齐
    if (tileLength == 0) {
        tileLength = 16;
    }

    uint32_t blockDim = (totalLength + tileLength - 1) / tileLength;
    blockDim = blockDim > 40 ? 40 : blockDim; // 限制核数上限（以实际硬件为准，如 910B 为 40/48）

    context->SetBlockDim(blockDim);
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

// shape 推导：输出 shape 与输入一致
static ge::graphStatus InferShapeFunc(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

// dtype 推导：输出为 float16
static ge::graphStatus InferDtypeFunc(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

// ---------------- 算子注册 ----------------
IMPL_OP(AddCustom)
    .InferShape(InferShapeFunc)
    .InferDataType(InferDtypeFunc)
    .Tiling(TilingFunc);

} // namespace ops
