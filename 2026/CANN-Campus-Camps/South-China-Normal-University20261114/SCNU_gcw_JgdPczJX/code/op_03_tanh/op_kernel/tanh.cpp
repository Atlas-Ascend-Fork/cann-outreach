#include "kernel_operator.h"
#include "tanh.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

using namespace AscendC;

template <uint32_t schMode>
__global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhTilingData);
    GET_TILING_DATA_WITH_STRUCT(TanhTilingData, tiling_data, tiling);
    
    if constexpr (schMode == TANH_TPL_SCH_MODE_0) {
        NsTanh::Tanh<float> op;
        op.Init(x, y, &tiling_data);
        op.Process();
    } else {
        NsTanh::Tanh<half> op;
        op.Init(x, y, &tiling_data);
        op.Process();
    }
}