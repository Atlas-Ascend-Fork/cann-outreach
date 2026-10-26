/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; 

template <typename DT_X>
class Sigmoid {
public:
    __aicore__ inline Sigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipe, const SigmoidTilingData* tilingData)
    {
        const int64_t blockIdx = GetBlockIdx();
        const int64_t blockOffset = static_cast<int64_t>(tilingData->blockLength) * blockIdx;
        int64_t remain = tilingData->totalLength - blockOffset;
        this->blockLength = remain < tilingData->blockLength ? remain : tilingData->blockLength;

        this->tileNum = static_cast<int32_t>(tilingData->tileNum);
        this->tileLength = static_cast<int32_t>(tilingData->tileLength);

        this->pipe = pipe;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + blockOffset, this->blockLength);
        this->pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        this->pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        this->pipe->InitBuffer(onesBuf, this->tileLength * sizeof(DT_X));
        this->onesLocal = onesBuf.Get<DT_X>();
        Duplicate(this->onesLocal, static_cast<DT_X>(1.0f), this->tileLength);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength <= 0) {
            return;
        }
        int32_t n = this->tileNum;
        const int32_t maxLoops = (this->blockLength + this->tileLength - 1) / this->tileLength;
        if (n > maxLoops) {
            n = maxLoops;
        }
        if (n <= 0) {
            return;
        }
        for (int32_t i = 0; i < n; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        const int32_t offset = progress * this->tileLength;
        this->currentLength = this->blockLength - offset;
        if (this->currentLength > this->tileLength) {
            this->currentLength = this->tileLength;
        }
        if (this->currentLength <= 0) {
            return;
        }
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        if (this->currentLength * static_cast<int32_t>(sizeof(DT_X)) % 32 == 0) {
            DataCopy(xLocal, xGm[offset], this->currentLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->currentLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        const int32_t len = this->currentLength;
        Muls(yLocal, xLocal, static_cast<DT_X>(-1.0f), len); 
        Exp(yLocal, yLocal, len);                            
        Adds(yLocal, yLocal, static_cast<DT_X>(1.0f), len);   
        Div(yLocal, this->onesLocal, yLocal, len);             
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        const int32_t offset = progress * this->tileLength;
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        if (this->currentLength * static_cast<int32_t>(sizeof(DT_X)) % 32 == 0) {
            DataCopy(yGm[offset], yLocal, this->currentLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->currentLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECIN> onesBuf;
    LocalTensor<DT_X> onesLocal;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    int32_t blockLength = 0;
    int32_t tileNum = 0;
    int32_t tileLength = 0;
    int32_t currentLength = 0;
};

} // namespace NsSigmoid
#endif // SIGMOID_H
