#include "tanh.h"

enum class TanhTilingKey : uint32_t {
    TILING_KEY_FP32 = 0,
    TILING_KEY_FP16 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhTilingData);
    GET_TILING_DATA_WITH_STRUCT(TanhTilingData, tilingData, tiling);

    if (schMode == static_cast<uint32_t>(TanhTilingKey::TILING_KEY_FP32)) {
        NsTanh::Tanh<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if (schMode == static_cast<uint32_t>(TanhTilingKey::TILING_KEY_FP16)) {
        NsTanh::Tanh<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
