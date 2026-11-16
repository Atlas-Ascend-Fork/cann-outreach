/*!
 * \file relu.cpp
 * \brief Compile separate streaming and one-shot ReLU implementations.
 */
#include "relu.h"

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    if constexpr (schMode == RELU_TPL_SCH_MODE_0) {
        NsRelu::Relu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == RELU_TPL_SCH_MODE_1) {
        NsRelu::Relu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == RELU_TPL_SCH_MODE_2) {
        AscendC::TPipe pipe;
        NsRelu::ReluSmall<half> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == RELU_TPL_SCH_MODE_3) {
        AscendC::TPipe pipe;
        NsRelu::ReluSmall<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}
