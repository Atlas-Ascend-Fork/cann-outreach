/*!
 * \file tanh.cpp
 * \brief Tanh 算子 kernel 入口
 */

#include "tanh.h"

__global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TanhTilingData);
    GET_TILING_DATA_WITH_STRUCT(TanhTilingData, tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        NsTanh::Tanh<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        NsTanh::Tanh<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}

