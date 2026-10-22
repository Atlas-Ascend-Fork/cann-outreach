/**
 * add_custom.cpp
 *
 * add_custom 算子 Host 侧实现，包含三部分：
 *   1. TilingFunc        —— 读取输入 shape，计算并写入 TilingData 与 numBlocks；
 *   2. InferShape/InferDataType —— 图模式下的输出 shape / dtype 推导（z 与输入 x 一致）；
 *   3. class AddCustom   —— 算子原型定义（输入 x/y、输出 z 的 dtype/format）与算子注册。
 *
 * 算子语义：z = x + y；数据类型 float16；数据格式 ND；shape 2 维 [N2, N1]。
 */
#include <cstddef>
#include <cstdint>

#include "../op_kernel/add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
// 核数：数据并行的 block 数量（整体数据按此数量切分到多个 AICore）
constexpr uint32_t NUM_BLOCKS = 8;
// 每个核上数据切分的块数（单核内流水分块）
constexpr uint32_t TILE_NUM = 8;

// Host 侧 Tiling 函数：框架在核函数启动前调用，写入核函数运行所需参数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取 TilingData 结构体指针（标准 C++ 结构体方式）
    AddCustomTilingData* tiling = context->GetTilingData<AddCustomTilingData>();
    // 读取第一个输入 x 的总元素数（2 维 shape [N2, N1] => N2 * N1）
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // 设置 SIMD 核数（对应核函数侧 GetBlockNum()）
    context->SetSimdNumBlocks(NUM_BLOCKS);

    // 显式写入核函数侧将要读取的全部字段
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    // 本算子不需要额外 workspace（未使用需要系统 workspace 的高阶 API）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
// 输出 shape 推导：z 与输入 x 形状完全一致（逐元素加法不改变形状）
static ge::graphStatus InferShape(ge::InferContext* context)
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ge::ShapePtr shapeX = context->GetInputShape(0);
    ge::ShapePtr shapeZ = context->GetOutputShape(0);
    if (shapeX == nullptr || shapeZ == nullptr) {
        GELOGE(0, "Get input or output shape failed.");
        return ge::GRAPH_FAILED;
    }
    const auto& originShapeX = shapeX->GetOriginShape();
    auto& originShapeZ = shapeZ->GetOriginShape();
    for (uint32_t i = 0; i < originShapeX.GetDimNum(); ++i) {
        originShapeZ.SetDimValue(i, originShapeX.GetDimValue(i));
    }
    originShapeZ.SetDimNum(originShapeX.GetDimNum());
    return ret;
}

// 输出 dtype 推导：z 与输入 x 数据类型一致
static ge::graphStatus InferDataType(ge::InferContext* context)
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ge::DataType dtypeX = context->GetInputDataType(0);
    ge::Tensor* tensorZ = context->GetOutputTensor(0);
    if (tensorZ == nullptr) {
        GELOGE(0, "Get output tensor failed.");
        return ge::GRAPH_FAILED;
    }
    tensorZ->SetDataType(dtypeX);
    return ret;
}

// 算子原型定义
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        // 输入 x：float16，ND
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        // 输入 y：float16，ND
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        // 输出 z：float16，ND
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 注册图模式下的输出 shape / dtype 推导函数
        this->SetInferShape(InferShape)
            .SetInferDataType(InferDataType);
        // 关联 Host 侧 Tiling 函数
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        // 注册支持的 AI 处理器型号（按实际芯片替换，如 ascend910b3 / ascend910_93 / 310b3 等）
        this->AICore().AddConfig("ascend910b3");
    }
};

// 结束算子注册
OP_ADD(AddCustom);
}  // namespace ops
