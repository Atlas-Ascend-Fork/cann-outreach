/*!
 * \file tanh.cpp
 * \brief Tanh 算子 kernel 入口
 */

#include "tanh.h"

template <uint32_t schMode>
__global__ __aicore__ void tanh(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        TanhTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        TanhTilingData,
        tilingData,
        tiling);

    /*
     * DTYPE_X 由 CANN 根据实际输入类型生成：
     *
     * float32 -> float
     * float16 -> half
     */
    NsTanh::Tanh<DTYPE_X> op;

    op.Init(
        x,
        y,
        &tilingData);

    op.Process();
}