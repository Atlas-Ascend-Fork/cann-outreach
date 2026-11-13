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
    
    // 引入临时 Buffer 用于存放中间结果
    TBuf<TPosition::VECCALC> tempBuf1;
    TBuf<TPosition::VECCALC> tempBuf2;
    TBuf<TPosition::VECCALC> tempBuf3;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t actualBlockLength_ = 0; // 当前核实际处理的数据量
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    
    int64_t offset = AscendC::GetBlockIdx() * blockLength_;
    actualBlockLength_ = tilingData->totalNum - offset;
    if (actualBlockLength_ > blockLength_) {
        actualBlockLength_ = blockLength_;
    } else if (actualBlockLength_ < 0) {
        actualBlockLength_ = 0;
    }

    if (actualBlockLength_ > 0) {
        inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, actualBlockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, actualBlockLength_);
    }

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    
    // 初始化临时 Buffer
    pipe.InitBuffer(tempBuf1, ubLength_ * sizeof(T)); // 存 exp(x)
    pipe.InitBuffer(tempBuf2, ubLength_ * sizeof(T)); // 存 exp(-x)
    pipe.InitBuffer(tempBuf3, ubLength_ * sizeof(T)); // 存中间结果
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    
    // 获取临时计算 Tensor
    LocalTensor<T> expX = tempBuf1.Get<T>();
    LocalTensor<T> expNegX = tempBuf2.Get<T>();
    LocalTensor<T> temp = tempBuf3.Get<T>();

    // 1. expX = exp(x)
    Exp(expX, xLocal, currentNum);

    // 2. expNegX = exp(-x) -> 先乘 -1 再 exp
    Muls(temp, xLocal, (T)-1.0, currentNum);
    Exp(expNegX, temp, currentNum);

    // 3. temp = exp(x) - exp(-x) (分子)
    Sub(temp, expX, expNegX, currentNum);

    // 4. expX = exp(x) + exp(-x) (分母)
    Add(expX, expX, expNegX, currentNum);

    // 5. y = 分子 / 分母
    Div(yLocal, temp, expX, currentNum);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (actualBlockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (actualBlockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = actualBlockLength_ - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H