/**
 * \file add_custom.cpp
 * \brief add_custom 算子 Host 侧实现：算子原型定义（OpDef）、Tiling 函数。
 *
 * 算子原型：
 *   - 输入 x、y：float16 / ND，2 维动态 shape [N2, N1]
 *   - 输出 z   ：float16 / ND，shape/dtype/format 跟随输入 x
 *   - 计算语义：z = x + y（逐元素加法）
 */
#include "../op_kernel/add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
// AICore 并行度控制参数
constexpr uint32_t MAX_BLOCK_DIM = 24;      // 单算子最多使用 24 个 AICore（910B 上限 40，此处留余量）
constexpr uint32_t MIN_LEN_PER_CORE = 2048; // 每核至少处理 2048 个元素，用于限制核数

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 取输入 x 的总元素数（N2 * N1）
    const uint32_t totalLength = static_cast<uint32_t>(context->GetInputTensor(0)->GetShapeSize());
    AddCustomTilingData *tiling = context->GetTilingData<AddCustomTilingData>();
    tiling->totalLength = totalLength;

    // 依据总元素数选择核数：每核至少 MIN_LEN_PER_CORE 个元素，且不超过 MAX_BLOCK_DIM
    uint32_t blockDim = (totalLength + MIN_LEN_PER_CORE - 1) / MIN_LEN_PER_CORE;
    if (blockDim > MAX_BLOCK_DIM)
    {
        blockDim = MAX_BLOCK_DIM;
    }
    if (blockDim < 1)
    {
        blockDim = 1;
    }
    tiling->blockDim = blockDim;
    context->SetBlockDim(blockDim);

    // 本算子不需要额外 GM workspace
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class AddCustom : public OpDef
{
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        // 输入 x：float16 / ND
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        // 输入 y：float16 / ND
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        // 输出 z：逐元素加法结果，shape/dtype/format 跟随输入 x
        this->Output("z")
            .ParamType(REQUIRED)
            .Follow("x");

        // 关联 Tiling 函数并注册本算子支持的昇腾 AI 处理器型号
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend310p");
    }
};
// 注册算子类型 AddCustom
OP_ADD(AddCustom);
}  // namespace ops
