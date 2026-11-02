/*!
 * \file sqrt.cpp
 * \brief Sqrt 算子 kernel 入口
 */

#include "sqrt.h"

template <typename DT_X>
__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA_WITH_STRUCT(SqrtTilingData, tilingData, tiling);
    NsSqrt::Sqrt<DT_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
