/*!
 * \file abs.cpp
 * \brief Abs 算子 kernel 入口
 */

#include "abs.h"

// CANN's datatype template arguments are emitted as numeric GE dtype IDs.
constexpr uint32_t ABS_FLOAT32 = 0;
constexpr uint32_t ABS_FLOAT16 = 1;

template <uint32_t DT_X>
__global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AbsTilingData);
    GET_TILING_DATA_WITH_STRUCT(AbsTilingData, tilingData, tiling);
    if constexpr (DT_X == ABS_FLOAT16) {
        NsAbs::Abs<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (DT_X == ABS_FLOAT32) {
        NsAbs::Abs<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
