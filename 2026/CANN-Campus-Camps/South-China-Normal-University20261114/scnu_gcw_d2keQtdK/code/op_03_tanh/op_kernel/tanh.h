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
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    
    // 申请一大块连续内存，用于切分
    TBuf<QuePosition::VECCALC> calcBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    totalLength_ = tilingData->totalNum;

    int64_t gmOffset = blockLength_ * AscendC::GetBlockIdx();
    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    
    // 按照 float 最大需求：xFloat, expPos, expNeg, tmpNum(暂存分子) 共计 4 份
    pipe.InitBuffer(calcBuf, ubLength_ * sizeof(float) * 4);
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

    // 从 calcBuf 中严格切分 4 块 float 内存，确保计算期间绝对不互相覆盖
    uint32_t offset = 0;
    uint32_t size = currentNum * sizeof(float);

    LocalTensor<float> expPos = calcBuf.GetWithOffset<float>(currentNum, offset); // 存 exp(x)
    offset += size;
    LocalTensor<float> expNeg = calcBuf.GetWithOffset<float>(currentNum, offset); // 存 exp(-x)
    offset += size;
    LocalTensor<float> xFloat = calcBuf.GetWithOffset<float>(currentNum, offset); // 存原始 x 的 float 版
    offset += size;
    LocalTensor<float> tmpNum = calcBuf.GetWithOffset<float>(currentNum, offset); // 存分子

    if constexpr (sizeof(T) == 4) {
        // float32 不需要转换，直接算
        Exp(expPos, xLocal, currentNum);                      // expPos = exp(x)
        Muls(expNeg, xLocal, -1.0f, currentNum);              // expNeg = -x
        Exp(expNeg, expNeg, currentNum);                      // expNeg = exp(-x)
        
        Sub(tmpNum, expPos, expNeg, currentNum);              // tmpNum = exp(x) - exp(-x)
        Add(expNeg, expPos, expNeg, currentNum);              // expNeg = exp(x) + exp(-x)
        Div(yLocal, tmpNum, expNeg, currentNum);              // y = tmpNum / expNeg
    } else {
        // float16 转 float32 计算，防止 Inf
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, currentNum); // xFloat = (float)x
        
        Exp(expPos, xFloat, currentNum);                      // expPos = exp(xFloat)
        Muls(expNeg, xFloat, -1.0f, currentNum);              // expNeg = -xFloat
        Exp(expNeg, expNeg, currentNum);                      // expNeg = exp(-xFloat)
        
        Sub(tmpNum, expPos, expNeg, currentNum);              // tmpNum = exp(x) - exp(-x)
        Add(expNeg, expPos, expNeg, currentNum);              // expNeg = exp(x) + exp(-x)
        
        Div(tmpNum, tmpNum, expNeg, currentNum);              // tmpNum = 结果 (float)
        
        Cast(yLocal, tmpNum, RoundMode::CAST_RINT, currentNum); // 转回 float16 存给 yLocal
    }

    outputQueueY.EnQue<T>(yLocal);
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
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t coreStart = blockIdx * blockLength_;
    int64_t currentBlockLength = blockLength_;
    
    // 边界截断
    if (coreStart + currentBlockLength > totalLength_) {
        currentBlockLength = totalLength_ - coreStart;
    }
    if (currentBlockLength <= 0) {
        return;
    }

    // 向上取整计算循环次数
    int64_t loopCount = (currentBlockLength + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (currentBlockLength - i * ubLength_) : ubLength_;
        if (currentNum <= 0) {
            break;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H