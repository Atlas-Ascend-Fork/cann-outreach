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
// float 计算过程的临时空间份数：2x、exp(2x)、分子/分母复用
constexpr int32_t TEMP_NUM = 3;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // 本核负责的元素数
    int64_t ubLength_ = 0;     // 单次 UB 循环处理的元素数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // 按核号切分全局数据，尾核处理剩余部分
    int64_t offset = AscendC::GetBlockIdx() * tilingData->blockFactor;
    int64_t currentBlockLength = tilingData->totalNum - offset;
    if (currentBlockLength > tilingData->blockFactor) {
        currentBlockLength = tilingData->blockFactor;
    }
    if (currentBlockLength < 0) {
        currentBlockLength = 0;
    }
    this->blockLength_ = currentBlockLength;
    this->ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuffer, TEMP_NUM * this->ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    int32_t count = static_cast<int32_t>(currentNum);
    int32_t tmpOffset = static_cast<int32_t>(this->ubLength_);
    LocalTensor<float> temp1 = tmpBuffer.Get<float>();
    LocalTensor<float> temp2 = temp1[tmpOffset];
    LocalTensor<float> temp3 = temp2[tmpOffset];

    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)，half 输入升到 float 计算后再转回，避免精度损失
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(temp1, xLocal, 2.0f, count);
        Exp(temp2, temp1, count);
        Adds(temp1, temp2, -1.0f, count);
        Adds(temp2, temp2, 1.0f, count);
        Div(yLocal, temp1, temp2, count);
    } else {
        Cast(temp1, xLocal, RoundMode::CAST_NONE, count);
        Muls(temp2, temp1, 2.0f, count);
        Exp(temp3, temp2, count);
        Adds(temp1, temp3, -1.0f, count);
        Adds(temp2, temp3, 1.0f, count);
        Div(temp3, temp1, temp2, count);
        Cast(yLocal, temp3, RoundMode::CAST_RINT, count);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t offset = 0;
    while (offset < this->blockLength_) {
        int64_t currentNum = this->blockLength_ - offset;
        if (currentNum > this->ubLength_) {
            currentNum = this->ubLength_;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
        offset += currentNum;
    }
}

} // namespace NsTanh
#endif // TANH_H
