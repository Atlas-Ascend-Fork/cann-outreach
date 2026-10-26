/*!
 * \file exp_infershape.cpp
 * \brief Exp 算子形状推导实现
 *
 * Exp 为一元逐元素算子（y = e^x），输出与输入形状完全一致，
 * 因此把输入 shape 原样拷贝到输出 shape 即可。
 * 注意：Exp 只有一个输入，若将来扩展为无输入算子，
 * input_shape 会为 nullptr，需要在此处手动设置输出 shape。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeExp(gert::InferShapeContext* context)
{
    // 取第 0 个输入的 shape 作为参考（Exp 只有输入 x）
    const gert::Shape* input_shape = (1 > 0) ? context->GetInputShape(0) : nullptr;
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* in_shape = (i < 1) ? context->GetInputShape(i) : input_shape;
        if (in_shape == nullptr) {
            in_shape = input_shape;
        }
        if (in_shape != nullptr) {
            // 一元算子：输出 shape = 输入 shape
            *output_shape = *in_shape;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Exp).InferShape(InferShapeExp);

} // namespace ops
