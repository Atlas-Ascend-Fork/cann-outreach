/*!
 * \file sigmoid_def.cpp
 * \brief Sigmoid 算子原型定义：y = 1 / (1 + exp(-x))
 */
#include "register/op_def_registry.h"

namespace ops {
class Sigmoid : public OpDef {
public:
    explicit Sigmoid(const char* name) : OpDef(name)
    {
        // 输入 x：必选，支持 float32 / float16，ND 布局，内存自动连续化
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输出 y：与输入同型同类型，必选，ND 布局，内存自动连续化
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 注册目标芯片：Ascend910B
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Sigmoid);
}  // namespace ops
