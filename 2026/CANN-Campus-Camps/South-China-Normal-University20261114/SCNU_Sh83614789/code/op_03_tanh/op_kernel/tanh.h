#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline void TanhCalc(const LocalTensor<float>& y, const LocalTensor<float>& x,
                                const LocalTensor<float>& tmp,
                                const LocalTensor<float>& t1, const LocalTensor<float>& t2, uint32_t n)
{
    Muls(t1, x, (float)2.0, n);
    Exp(t1, t1, n);
    Adds(t2, t1, (float)1.0, n);
    Duplicate(tmp, (float)2.0, n);
    Div(tmp, tmp, t2, n);
    Muls(tmp, tmp, (float)(-1.0), n);
    Adds(y, tmp, (float)1.0, n);
}

__aicore__ inline void TanhCalc(const LocalTensor<half>& y, const LocalTensor<half>& x,
                                const LocalTensor<float>& tmp,
                                const LocalTensor<float>& t1, const LocalTensor<float>& t2, uint32_t n)
{
    Cast(tmp, x, RoundMode::CAST_NONE, n);
    Muls(t1, tmp, (float)2.0, n);
    Exp(t1, t1, n);
    Adds(t2, t1, (float)1.0, n);
    Duplicate(tmp, (float)2.0, n);
    Div(tmp, tmp, t2, n);
    Muls(tmp, tmp, (float)(-1.0), n);
    Adds(tmp, tmp, (float)1.0, n);
    Cast(y, tmp, RoundMode::CAST_RINT, n);
}

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], this->tileLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        LocalTensor<float> fTmp = tmpBuf0.Get<float>();
        LocalTensor<float> f1 = tmpBuf1.Get<float>();
        LocalTensor<float> f2 = tmpBuf2.Get<float>();
        TanhCalc(yLocal, xLocal, fTmp, f1, f2, this->tileLength);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileLength], yLocal, this->tileLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tmpBuf0;
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsTanh
#endif // TANH_H
