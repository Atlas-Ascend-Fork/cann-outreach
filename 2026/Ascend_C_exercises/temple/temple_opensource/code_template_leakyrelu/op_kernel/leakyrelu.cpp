/*!
 * \file leakyrelu.cpp
 * \brief Leakyrelu 算子 kernel 入口
 */

#include "leakyrelu.h"

template <typename DT_X>
__global__ __aicore__ void leakyrelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LeakyreluTilingData);
    GET_TILING_DATA_WITH_STRUCT(LeakyreluTilingData, tilingData, tiling);
    // TODO 考生自行补齐
}
