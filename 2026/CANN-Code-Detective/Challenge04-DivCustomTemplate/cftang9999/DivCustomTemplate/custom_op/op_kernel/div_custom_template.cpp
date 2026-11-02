#include "kernel_operator.h"
#include "div_custom_template_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

class KernelDivCustomTemplate {
public:
    __aicore__ inline KernelDivCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, DivCustomTemplateTilingData *tiling) {
        totalLength = tiling->totalLength;
        tileNum = tiling->tileNum;
        tileLength = tiling->tileLength;

        xGm.SetGlobalBuffer((__gm__ half *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ half *)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ half *)z, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (tileNum == 0) return;
        CopyIn(0);
        for (int32_t i = 0; i < (int32_t)tileNum - 1; i++) {
            CopyIn(i + 1);
            Compute(i);
            CopyOut(i);
        }
        Compute(tileNum - 1);
        CopyOut(tileNum - 1);
    }

private:
    __aicore__ inline uint32_t CalcTileLength(int32_t progress) const {
        if (progress == (int32_t)tileNum - 1) {
            return totalLength - progress * tileLength;
        }
        return tileLength;
    }

    __aicore__ inline void CopyIn(int32_t progress) {
        uint32_t cur = CalcTileLength(progress);
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * tileLength], cur);
        DataCopy(yLocal, yGm[progress * tileLength], cur);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        uint32_t cur = CalcTileLength(progress);
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        uint32_t calcCount = (cur + 7) / 8 * 8;
        Div(zLocal, xLocal, yLocal, calcCount);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        uint32_t cur = CalcTileLength(progress);
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * tileLength], zLocal, cur);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DivCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelDivCustomTemplate op;
    op.Init(x, y, z, &tilingData);
    op.Process();
}
