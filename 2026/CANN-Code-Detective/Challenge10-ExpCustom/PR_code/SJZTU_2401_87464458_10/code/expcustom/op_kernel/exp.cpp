/*!
 * \file exp.cpp
 * \brief Exp operator kernel entry.
 */

#include "exp.h"

template <typename DT_X>
__global__ __aicore__ void exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ExpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ExpTilingData, tilingData, tiling);
    NsExp::Exp<DT_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
