#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

#define DTYPE_X half
#define DTYPE_Y half

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->tileNum = tileNum;
        this->blockLength = totalLength / tileNum;          // elements per tile-pair
        this->tileLength = this->blockLength / BUFFER_NUM;  // elements per tile
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, totalLength);
        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // alloc tensor from queue memory
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        // copy progress_th tile from GM to VECIN
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // temp buffers for exp(x), exp(-x) and intermediate results
        AscendC::LocalTensor<DTYPE_X> expPos = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNeg = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuf2.Get<DTYPE_X>();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::Exp(expPos, xLocal, tileLength);        // exp(x)
        AscendC::Muls(tmp, xLocal, static_cast<DTYPE_X>(-1.0f), tileLength); // -x
        AscendC::Exp(expNeg, tmp, tileLength);           // exp(-x)
        AscendC::Sub(tmp, expPos, expNeg, tileLength);   // exp(x) - exp(-x)
        AscendC::Add(expNeg, expPos, expNeg, tileLength);// exp(x) + exp(-x)
        AscendC::Div(yLocal, tmp, expNeg, tileLength);   // (e^x - e^-x) / (e^x + e^-x)

        // enque the output tensor to VECOUT queue
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // deque output tensor from VECOUT queue
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        // copy progress_th tile from VECOUT to GM
        AscendC::DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        // free output tensor for reuse
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
