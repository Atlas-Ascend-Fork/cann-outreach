/*!
 * \file sqrt.cpp
 * \brief Sqrt 算子核函数入口
 */

#include "sqrt.h"

template <uint32_t calcMode>
__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA_WITH_STRUCT(SqrtTilingData, tilingData, tiling);
    if constexpr (calcMode == SQRT_CALC_MODE_FP16_UPCAST) {
        NsSqrt::SqrtTile<half, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (calcMode == SQRT_CALC_MODE_FP32) {
        NsSqrt::SqrtTile<float, false> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}