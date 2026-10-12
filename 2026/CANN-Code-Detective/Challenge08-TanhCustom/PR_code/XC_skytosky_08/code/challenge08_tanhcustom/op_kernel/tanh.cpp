/*!
 * \file tanh.cpp
 * \brief Tanh 算子 kernel 入口
 */

#include "tanh.h"

template <int DT_X>
__global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhTilingData);
    GET_TILING_DATA_WITH_STRUCT(TanhTilingData, tilingData, tiling);
    // TilingKey 模板参数为模板组合的序号：0 对应 C_DT_FLOAT，1 对应 C_DT_FLOAT16
    if (DT_X == 0) {
        NsTanh::Tanh<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else {
        NsTanh::Tanh<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
