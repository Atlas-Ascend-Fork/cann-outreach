#include "kernel_operator.h"

#include "tanh.h"

using namespace AscendC;
using namespace tanh_op;

extern "C" __global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ const uint32_t* td = reinterpret_cast<__gm__ const uint32_t*>(tiling);

    KernelTanh<DTYPE_X> op;
    op.Init(x, y, td[0], td[1], td[3], td[4]);
    op.Process();
}