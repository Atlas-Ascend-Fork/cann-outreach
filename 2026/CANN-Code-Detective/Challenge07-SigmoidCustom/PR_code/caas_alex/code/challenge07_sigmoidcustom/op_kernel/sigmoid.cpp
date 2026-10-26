/*!
 * \file sigmoid.cpp
 * \brief Sigmoid 算子 kernel 入口
 */

#include "sigmoid.h"

using namespace AscendC;

template <typename DT_X>
__global__ __aicore__ void sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidTilingData);
    GET_TILING_DATA_WITH_STRUCT(SigmoidTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    NsSigmoid::Sigmoid<DT_X> op;
    op.Init(x, y, &pipe, &tilingData);
    op.Process();
}
