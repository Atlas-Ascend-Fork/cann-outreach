#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tmpBufSrc;
    TBuf<TPosition::VECCALC> tmpBufExpX;
    TBuf<TPosition::VECCALC> tmpBufExpNegX;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    this->ubLength_ = tilingData->ubFactor;

    constexpr int64_t MAX_UB_BYTES = 192 * 1024;
    constexpr int64_t BLOCK_PER_CORE = 2 * BUFFER_NUM + 3;
    int64_t maxElems = MAX_UB_BYTES / BLOCK_PER_CORE / static_cast<int64_t>(sizeof(float));
    if (this->ubLength_ > maxElems) {
        this->ubLength_ = maxElems;
    }
    const int64_t blockFactor = tilingData->blockFactor;
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t startOffset = blockIdx * blockFactor;

    int64_t currentLength = tilingData->totalNum - startOffset;
    if (currentLength > blockFactor) {
        currentLength = blockFactor;
    }
    if (currentLength <= 0) {
        this->blockLength_ = 0;
        return;
    }
    this->blockLength_ = currentLength;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(this->ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(this->ubLength_ * sizeof(T)));
    pipe.InitBuffer(tmpBufSrc, static_cast<uint32_t>(this->ubLength_ * sizeof(float)));
    pipe.InitBuffer(tmpBufExpX, static_cast<uint32_t>(this->ubLength_ * sizeof(float)));
    pipe.InitBuffer(tmpBufExpNegX, static_cast<uint32_t>(this->ubLength_ * sizeof(float)));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * this->ubLength_], copyParams, padParams);

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> srcLocal = tmpBufSrc.Get<float>();
    LocalTensor<float> expXLocal = tmpBufExpX.Get<float>();
    LocalTensor<float> expNegXLocal = tmpBufExpNegX.Get<float>();
    const int32_t count = static_cast<int32_t>(currentNum);

    if constexpr (sizeof(T) == sizeof(float)) {
        Exp(expXLocal, xLocal, count);
        Muls(expNegXLocal, xLocal, (float)-1.0, count);
        Exp(expNegXLocal, expNegXLocal, count);
        Sub(yLocal, expXLocal, expNegXLocal, count);
        Add(expXLocal, expXLocal, expNegXLocal, count);
        Div(yLocal, yLocal, expXLocal, count);
    } else {
        Cast(srcLocal, xLocal, RoundMode::CAST_NONE, count);
        Exp(expXLocal, srcLocal, count);
        Muls(expNegXLocal, srcLocal, (float)-1.0, count);
        Exp(expNegXLocal, expNegXLocal, count);
        Sub(srcLocal, expXLocal, expNegXLocal, count);
        Add(expXLocal, expXLocal, expNegXLocal, count);
        Div(srcLocal, srcLocal, expXLocal, count);
        Cast(yLocal, srcLocal, RoundMode::CAST_RINT, count);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * this->ubLength_], yLocal, copyParams);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (this->blockLength_ <= 0 || this->ubLength_ <= 0) {
        return;
    }

    int64_t tileNum = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    int64_t lastNum = this->blockLength_ - (tileNum - 1) * this->ubLength_;

    CopyIn(0, (tileNum == 1) ? lastNum : this->ubLength_);
    for (int64_t i = 0; i < tileNum; i++) {
        int64_t curNum = (i == tileNum - 1) ? lastNum : this->ubLength_;

        if (i + 1 < tileNum) {
            int64_t nextNum = ((i + 1) == tileNum - 1) ? lastNum : this->ubLength_;
            CopyIn(i + 1, nextNum);
        }

        Compute(curNum);
        CopyOut(i, curNum);
    }
}

}

#endif