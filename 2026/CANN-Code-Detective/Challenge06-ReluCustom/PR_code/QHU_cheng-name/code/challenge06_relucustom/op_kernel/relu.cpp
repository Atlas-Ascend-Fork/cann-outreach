/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

template <uint32_t tilingKey>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    if constexpr (tilingKey == 0) {
        // float32 实例（框架实例序号 0 = C_DT_FLOAT）
        NsRelu::Relu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else {
        // float16 实例（框架实例序号 1 = C_DT_FLOAT16）
        NsRelu::Relu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
