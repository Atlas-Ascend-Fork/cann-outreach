/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

// 非模板核函数：fp32/fp16 在核内按 tiling 中的 dtype 运行时分发
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    if (tilingData.dtype == 1) {
        // fp16
        NsRelu::Relu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else {
        // fp32
        NsRelu::Relu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}