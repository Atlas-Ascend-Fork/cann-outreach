/*!
 * \file relu_def.cpp
 * \brief Relu 算子定义
 *
 * 说明：本文件只做算子原型注册（OP_ADD），支持 ND 格式与 float32 / float16。
 *   Tiling 注册只允许出现在 op_host/relu_tiling.cpp 一处，两处都写会链接报
 *   multiple definition of `optiling::op_impl_register_optiling_Relu`。
 */
#include "register/op_def_registry.h"

namespace ops {
class Relu : public OpDef {
public:
    explicit Relu(const char* name) : OpDef(name)
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

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Relu);
} // namespace ops