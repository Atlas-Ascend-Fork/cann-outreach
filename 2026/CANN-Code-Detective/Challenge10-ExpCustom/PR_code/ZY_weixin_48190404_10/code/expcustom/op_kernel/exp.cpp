/*!
 * \file exp.cpp
 * \brief Exp 算子 kernel 入口
 */

#include "exp.h"

// 使用 schMode 作为模板参数，对应 exp_tiling_key.h 里的定义
template <uint32_t schMode>
__global__ __aicore__ void exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ExpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ExpTilingData, tilingData, tiling);

    // 对应 EXP_TPL_SCH_MODE_0 (Half)
    if constexpr (schMode == EXP_TPL_SCH_MODE_0) {
        NsExp::Exp<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    // 对应 EXP_TPL_SCH_MODE_1 (Float)
    if constexpr (schMode == EXP_TPL_SCH_MODE_1) {
        NsExp::Exp<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}