/*!
 * \file exp_def.cpp
 * \brief Exp 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class Exp : public OpDef {
public:
    explicit Exp(const char* name) : OpDef(name)
    {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

        // 同时声明 ascend910b 与 ascend910_93，覆盖 A2 训练/推理产品线
        // （回应 PR 检视意见，保留 ascend910_93 配置）。
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Exp);
} // namespace ops
