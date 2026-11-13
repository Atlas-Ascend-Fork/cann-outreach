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

#include <type_traits>

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

    // 计算过程所需临时变量：float32 中间量与 exp 中间结果
    TBuf<TPosition::VECCALC> tmpXBuffer;  // float16 输入转 float32 后的缓存
    TBuf<TPosition::VECCALC> tmpBuffer1;  // e^x
    TBuf<TPosition::VECCALC> tmpBuffer2;  // e^-x

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // 本核处理的起始偏移与元素数量（尾核可能不满 blockFactor）
    int64_t startOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    blockLength_ = tilingData->blockFactor;
    if (startOffset + blockLength_ > tilingData->totalNum) {
        blockLength_ = tilingData->totalNum - startOffset;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    // float32 中间计算所需临时缓存
    pipe.InitBuffer(tmpXBuffer, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuffer1, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuffer2, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y = tanh(x) = (e^x - e^-x) / (e^x + e^-x)，用 float32 中间计算保证精度
    if constexpr (std::is_same<T, half>::value) {
        LocalTensor<float> xFloat = tmpXBuffer.Get<float>();
        LocalTensor<float> expPos = tmpBuffer1.Get<float>();
        LocalTensor<float> expNeg = tmpBuffer2.Get<float>();
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(currentNum));
        Exp(expPos, xFloat, static_cast<uint32_t>(currentNum));          // e^x
        Muls(expNeg, xFloat, -1.0f, static_cast<uint32_t>(currentNum));  // -x
        Exp(expNeg, expNeg, static_cast<uint32_t>(currentNum));          // e^-x
        Sub(xFloat, expPos, expNeg, static_cast<uint32_t>(currentNum));  // e^x - e^-x
        Add(expPos, expPos, expNeg, static_cast<uint32_t>(currentNum));  // e^x + e^-x
        Div(xFloat, xFloat, expPos, static_cast<uint32_t>(currentNum));  // tanh(x)
        Cast(yLocal, xFloat, RoundMode::CAST_RINT, static_cast<uint32_t>(currentNum));
    } else {
        LocalTensor<float> expPos = tmpBuffer1.Get<float>();
        LocalTensor<float> expNeg = tmpBuffer2.Get<float>();
        Exp(expPos, xLocal, static_cast<uint32_t>(currentNum));          // e^x
        Muls(expNeg, xLocal, -1.0f, static_cast<uint32_t>(currentNum));  // -x
        Exp(expNeg, expNeg, static_cast<uint32_t>(currentNum));          // e^-x
        Sub(yLocal, expPos, expNeg, static_cast<uint32_t>(currentNum));  // e^x - e^-x
        Add(expPos, expPos, expNeg, static_cast<uint32_t>(currentNum));  // e^x + e^-x
        Div(yLocal, yLocal, expPos, static_cast<uint32_t>(currentNum));  // tanh(x)
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = ubLength_;
        if (i == loopCount - 1 && blockLength_ % ubLength_ != 0) {
            currentNum = blockLength_ - i * ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
