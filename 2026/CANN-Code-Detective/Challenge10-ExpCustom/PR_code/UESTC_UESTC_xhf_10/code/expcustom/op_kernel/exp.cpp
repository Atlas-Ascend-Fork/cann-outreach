/*!
 * \file exp.cpp
 * \brief Exp 算子 kernel 入口
 */

#include "exp.h"

template <uint32_t schMode>
__global__ __aicore__ void exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ExpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ExpTilingData, tilingData, tiling);
    if constexpr (schMode == EXP_TPL_SCH_MODE_0) {
        NsExp::Exp<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == EXP_TPL_SCH_MODE_1) {
        NsExp::Exp<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
