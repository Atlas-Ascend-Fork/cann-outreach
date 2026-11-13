/*!
 * \file abs.cpp
 * \brief Abs 算子 kernel 入口
 */

#include "abs.h"

// schMode 为 tilingKey 模板实例编号：ABS_TPL_SCH_MODE_0 -> half，ABS_TPL_SCH_MODE_1 -> float
template <uint32_t schMode>
__global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AbsTilingData);
    GET_TILING_DATA_WITH_STRUCT(AbsTilingData, tilingData, tiling);
    if constexpr (schMode == ABS_TPL_SCH_MODE_0) {
        NsAbs::Abs<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == ABS_TPL_SCH_MODE_1) {
        NsAbs::Abs<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
