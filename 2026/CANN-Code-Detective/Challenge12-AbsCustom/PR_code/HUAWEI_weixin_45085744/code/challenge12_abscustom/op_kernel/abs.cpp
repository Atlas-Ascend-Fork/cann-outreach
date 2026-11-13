/*!
 * \file abs.cpp
 * \brief Abs 算子 kernel 入口
 */

#include "abs.h"

template <typename DT_X>
__global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AbsTilingData);
    GET_TILING_DATA_WITH_STRUCT(AbsTilingData, tilingData, tiling);
    NsAbs::Abs<DT_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
