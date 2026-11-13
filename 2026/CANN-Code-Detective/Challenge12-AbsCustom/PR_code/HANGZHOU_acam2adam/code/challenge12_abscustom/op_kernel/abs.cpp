/*!
 * \file abs.cpp
 * \brief Abs 算子 kernel 入口（专用 shape 8x2048 = 16384）
 */

#include "abs.h"

template <typename DT_X>
__global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    (void)tiling;
    REGISTER_TILING_DEFAULT(AbsTilingData);
    NsAbs::Abs<DT_X> op;
    op.Init(x, y);
    op.Process();
}
