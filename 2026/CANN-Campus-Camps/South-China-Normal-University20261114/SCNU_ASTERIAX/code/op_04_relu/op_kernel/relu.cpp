#include "relu.h"
template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (schMode == 2) {
        NsRelu::SmallRelu<half, 4096>(x, y);
    } else if constexpr (schMode == 3) {
        NsRelu::SmallRelu<float, 4096>(x, y);
    } else if constexpr (schMode == 4) {
        NsRelu::LargeRelu<half>(x, y);
    } else if constexpr (schMode == 5) {
        NsRelu::LargeRelu<float>(x, y);
    } else {
        REGISTER_TILING_DEFAULT(ReluTilingData);
        GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
        if constexpr (schMode == 0) {
            NsRelu::Relu<half> op;
            op.Init(x, y, &tilingData);
            op.Process();
        } else if constexpr (schMode == 1) {
            NsRelu::Relu<float> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
    }
}
