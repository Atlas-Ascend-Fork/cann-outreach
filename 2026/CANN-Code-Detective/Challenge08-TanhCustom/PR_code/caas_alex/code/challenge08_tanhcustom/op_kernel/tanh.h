/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipe, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int32_t progress);

private:
    TPipe* pipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECIN> negTwoBuf;
    LocalTensor<DT_X> negTwoLocal;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    int32_t blockLength = 0;
    int32_t ubFactor = 0;
    int32_t currentLength = 0;
};

template <typename DT_X>
__aicore__ inline void Tanh<DT_X>::Init(GM_ADDR x, GM_ADDR y, TPipe* pipe, const TanhTilingData* tilingData)
{
    const int64_t blockIdx = GetBlockIdx();
    const int64_t blockOffset = static_cast<int64_t>(tilingData->blockFactor) * blockIdx;
    int64_t remain = tilingData->totalNum - blockOffset;
    this->blockLength = remain < tilingData->blockFactor ? remain : tilingData->blockFactor;

    this->ubFactor = static_cast<int32_t>(tilingData->ubFactor);

    this->pipe = pipe;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + blockOffset, this->blockLength);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + blockOffset, this->blockLength);
    this->pipe->InitBuffer(inQueueX, BUFFER_NUM, this->ubFactor * sizeof(DT_X));
    this->pipe->InitBuffer(outQueueY, BUFFER_NUM, this->ubFactor * sizeof(DT_X));
    this->pipe->InitBuffer(negTwoBuf, this->ubFactor * sizeof(DT_X));
    this->negTwoLocal = negTwoBuf.Get<DT_X>();
    Duplicate(this->negTwoLocal, static_cast<DT_X>(-2.0f), this->ubFactor);
    PipeBarrier<PIPE_V>();
}

template <typename DT_X>
__aicore__ inline void Tanh<DT_X>::Process()
{
    if (this->blockLength <= 0) {
        return;
    }
    const int32_t n = (this->blockLength + this->ubFactor - 1) / this->ubFactor;
    CopyIn(0);
    for (int32_t i = 0; i < n - 1; i++) {
        Compute();
        CopyOut(i);
        CopyIn(i + 1);
    }
    Compute();
    CopyOut(n - 1);
}

template <typename DT_X>
__aicore__ inline void Tanh<DT_X>::CopyIn(int32_t progress)
{
    const int32_t offset = progress * this->ubFactor;
    this->currentLength = this->blockLength - offset;
    if (this->currentLength > this->ubFactor) {
        this->currentLength = this->ubFactor;
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

template <typename DT_X>
__aicore__ inline void Tanh<DT_X>::Compute()
{
    LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
    LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
    const int32_t len = this->currentLength;
    Muls(yLocal, xLocal, static_cast<DT_X>(2.0f), len);
    Exp(yLocal, yLocal, len);                          
    Adds(yLocal, yLocal, static_cast<DT_X>(1.0f), len);
    Div(yLocal, this->negTwoLocal, yLocal, len);       
    Adds(yLocal, yLocal, static_cast<DT_X>(1.0f), len);
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename DT_X>
__aicore__ inline void Tanh<DT_X>::CopyOut(int32_t progress)
{
    const int32_t offset = progress * this->ubFactor;
    LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
    if (this->currentLength * static_cast<int32_t>(sizeof(DT_X)) % 32 == 0) {
        DataCopy(yGm[offset], yLocal, this->currentLength);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->currentLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
    }
    outQueueY.FreeTensor(yLocal);
}

} // namespace NsTanh
#endif // TANH_H
