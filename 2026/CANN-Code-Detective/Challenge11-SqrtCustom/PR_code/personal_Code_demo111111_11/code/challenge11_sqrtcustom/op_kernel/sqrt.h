/*!
 * \file sqrt.h
 * \brief Sqrt 算子 kernel 类定义
 */

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace NsSqrt {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Sqrt {
public:
    __aicore__ inline Sqrt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    // 用于 half 升精度计算的临时 TBuf
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Sqrt<T>::Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t blockStart = blockIdx * tilingData->blockFactor;

    if (blockStart > tilingData->totalNum) {
        blockStart = tilingData->totalNum;
    }

    if (blockStart >= tilingData->totalNum || tilingData->ubFactor <= 0) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }

    int64_t remain = tilingData->totalNum - blockStart;
    blockLength_ = tilingData->blockFactor < remain ? tilingData->blockFactor : remain;
    
    ubLength_ = tilingData->ubFactor;
    // 【防越界保护】无论外部传入的 ubFactor 多大，内部统一限制在 8192 内
    // 防止 test_sqrt.cpp 中硬编码 ubFactor=16384 导致 UB 内存爆掉
    constexpr int64_t MAX_UB_LEN = 8192;
    if (ubLength_ > MAX_UB_LEN) {
        ubLength_ = MAX_UB_LEN;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockStart, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockStart, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));

    // 如果是 half 类型，申请额外的 float 升精度空间
    if (sizeof(T) != 4) {
        pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    
    // 【修改点】使用 DataCopyPad 支持尾部非 32B 对齐的数据搬运，防止越界读脏数据
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(inputLocal, inputGMX[progress * ubLength_], copyParams, DataCopyPadExtParams<T>{});
    
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    if (sizeof(T) == 4) {
        // float 直接计算
        AscendC::Sqrt(outputLocal, inputLocal, currentNum);
    } else {
        // half 升精度计算，避免精度不够导致 Wrong Answer
        LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
        Cast(tmpFloat, inputLocal, RoundMode::CAST_NONE, currentNum);
        AscendC::Sqrt(tmpFloat, tmpFloat, currentNum);
        // 使用 CAST_RINT 四舍五入回 half
        Cast(outputLocal, tmpFloat, RoundMode::CAST_RINT, currentNum);
    }

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    
    // 【修改点】使用 DataCopyPad 支持尾部非 32B 对齐的数据写出，防止越界写
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGMY[progress * ubLength_], outputLocal, copyParams);
    
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = ubLength_;
        if (i == loopCount - 1) {
            currentNum = blockLength_ - i * ubLength_;
        }

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSqrt
#endif // SQRT_H