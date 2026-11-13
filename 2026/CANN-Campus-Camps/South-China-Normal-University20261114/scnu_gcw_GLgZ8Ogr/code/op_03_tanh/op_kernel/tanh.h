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
    TBuf<TPosition::VECCALC> tmpBuffer;   // 计算 tanh 所需临时 buffer（存放 e^x、e^{-x}）

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
};

// 初始化：按核切分 GM，并按 ubFactor 分配 UB
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    totalNum_ = tilingData->totalNum;

    // 每个核处理 [blockIdx * blockLength, ...) 区间的数据
    int64_t offset = blockLength_ * static_cast<int64_t>(GetBlockIdx());
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // 输入/输出队列各 double buffer，另加 1 块计算临时 buffer
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(tmpBuffer, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], currentNum);
    inputQueueX.EnQue(xLocal);
}

// tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> tmpLocal = tmpBuffer.Get<T>();

    Exp(yLocal, xLocal, currentNum);                                  // y = e^x
    Muls(xLocal, xLocal, static_cast<T>(-1), currentNum);             // x = -x
    Exp(xLocal, xLocal, currentNum);                                  // x = e^{-x}
    Sub(tmpLocal, yLocal, xLocal, currentNum);                        // tmp = e^x - e^{-x}
    Add(yLocal, yLocal, xLocal, currentNum);                          // y   = e^x + e^{-x}
    Div(yLocal, tmpLocal, yLocal, currentNum);                        // y   = tanh(x)

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // 最后一个核实际处理的数据量可能小于 blockLength_
    int64_t currentNum = totalNum_ - blockLength_ * static_cast<int64_t>(GetBlockIdx());
    if (currentNum > blockLength_) {
        currentNum = blockLength_;
    }
    if (currentNum <= 0) {
        return;
    }

    int64_t loopCount = (currentNum + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t offset = i * ubLength_;
        int64_t num = (currentNum - offset < ubLength_) ? (currentNum - offset) : ubLength_;
        CopyIn(offset, num);
        Compute(num);
        CopyOut(offset, num);
    }
}

} // namespace NsTanh
#endif // TANH_H
