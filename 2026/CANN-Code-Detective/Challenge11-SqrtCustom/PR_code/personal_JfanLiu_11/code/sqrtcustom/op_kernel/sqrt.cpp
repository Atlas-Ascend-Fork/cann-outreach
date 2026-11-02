/*!
 * \file sqrt.cpp
 * \brief Sqrt 算子核函数入口
 */

#include "sqrt.h"

template <uint32_t schMode>
__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA_WITH_STRUCT(SqrtTilingData, tilingData, tiling);
    if constexpr (schMode == SQRT_SCH_MODE_FP16) {
        NsSqrt::SqrtVec<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == SQRT_SCH_MODE_FP32) {
        NsSqrt::SqrtVec<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}