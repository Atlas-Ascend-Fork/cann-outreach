/*!
 * \file leakyrelu_def.cpp
 * \brief Leakyrelu 算子定义（本系列首个带属性 negative_slope 的算子）
 */
#include "register/op_def_registry.h"

namespace ops {
class Leakyrelu : public OpDef {
public:
    explicit Leakyrelu(const char* name) : OpDef(name)
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

    // 属性：负半轴斜率 α（float，必选），参考 torch.nn.functional.leaky_relu 的 negative_slope
    this->Attr("negative_slope")
        .FloatAttr(REQUIRED);

    this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Leakyrelu);
} // namespace ops
