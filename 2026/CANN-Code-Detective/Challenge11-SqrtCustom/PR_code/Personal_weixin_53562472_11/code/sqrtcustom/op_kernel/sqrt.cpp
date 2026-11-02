/*!
 * \file sqrt.cpp
 * \brief Sqrt 算子 kernel 入口
 */

#include "sqrt.h"

// schMode 为 tilingKey 模板实例编号：SQRT_TPL_SCH_MODE_0 -> half，SQRT_TPL_SCH_MODE_1 -> float
template <uint32_t schMode>
__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA_WITH_STRUCT(SqrtTilingData, tilingData, tiling);
    if constexpr (schMode == SQRT_TPL_SCH_MODE_0) {
        NsSqrt::Sqrt<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == SQRT_TPL_SCH_MODE_1) {
        NsSqrt::Sqrt<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
