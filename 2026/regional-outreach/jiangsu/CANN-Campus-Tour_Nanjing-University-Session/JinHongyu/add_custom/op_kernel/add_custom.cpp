/*!
 * \file add_custom.cpp
 * \brief add_custom 算子 kernel 入口
 */

#include "add_custom.h"

template <typename DT_X>
__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTilingData, tilingData, tiling);
    NsAddCustom::AddCustom<DT_X> op;
    op.Init(x, y, z, &tilingData);
    op.Process();
}
