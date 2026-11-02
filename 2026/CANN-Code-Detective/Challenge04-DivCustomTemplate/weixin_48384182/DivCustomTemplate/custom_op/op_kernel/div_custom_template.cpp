#include "kernel_operator.h"

using namespace AscendC;

#define DTYPE_X half
#define DTYPE_Y half
#define DTYPE_Z half

constexpr int32_t BUFFER_NUM = 2;

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t totalLength = tiling_data.size;

    uint32_t coreNum = 8;
    uint32_t coreId = GetBlockIdx();

    uint32_t perCore = (totalLength + coreNum - 1) / coreNum;
    uint32_t start = coreId * perCore;
    uint32_t end = start + perCore;
    if (end > totalLength) {
        end = totalLength;
    }
    if (start >= end) {
        return;
    }
    uint32_t curLength = end - start;

    GlobalTensor<DTYPE_X> xGlobal;
    GlobalTensor<DTYPE_Y> yGlobal;
    GlobalTensor<DTYPE_Z> zGlobal;
    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(x), totalLength);
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(y), totalLength);
    zGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Z *>(z), totalLength);

    TQue<QuePosition::VECIN, BUFFER_NUM> queX, queY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queZ;

    TPipe pipe;
    pipe.InitBuffer(queX, BUFFER_NUM, perCore * sizeof(DTYPE_X));
    pipe.InitBuffer(queY, BUFFER_NUM, perCore * sizeof(DTYPE_Y));
    pipe.InitBuffer(queZ, BUFFER_NUM, perCore * sizeof(DTYPE_Z));

    LocalTensor<DTYPE_X> xLocal = queX.AllocTensor<DTYPE_X>();
    LocalTensor<DTYPE_Y> yLocal = queY.AllocTensor<DTYPE_Y>();
    LocalTensor<DTYPE_Z> zLocal = queZ.AllocTensor<DTYPE_Z>();

    DataCopy(xLocal, xGlobal[start], curLength);
    DataCopy(yLocal, yGlobal[start], curLength);
    queX.EnQue(xLocal);
    queY.EnQue(yLocal);

    LocalTensor<DTYPE_X> xCalc = queX.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_Y> yCalc = queY.DeQue<DTYPE_Y>();

    Div(zLocal, xCalc, yCalc, curLength);

    queZ.EnQue(zLocal);
    LocalTensor<DTYPE_Z> zOut = queZ.DeQue<DTYPE_Z>();
    DataCopy(zGlobal[start], zOut, curLength);

    queX.FreeTensor(xCalc);
    queY.FreeTensor(yCalc);
    queZ.FreeTensor(zOut);
}
