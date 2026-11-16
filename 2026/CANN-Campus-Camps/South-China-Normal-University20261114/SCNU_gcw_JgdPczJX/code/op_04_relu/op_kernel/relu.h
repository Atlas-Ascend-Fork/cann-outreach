/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    // 计算当前核在全局内存的起始偏移
    int64_t startOffset = GetBlockIdx() * blockLength_;
    
    // 防御性处理：如果起始偏移越界，将其重置为 0，防止构造非法 GlobalTensor
    if (startOffset >= totalNum_) {
        startOffset = 0;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset);

    // 分配 UB 双缓冲队列
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // ReLU 计算逻辑：y = max(0, x)
    Maxs(yLocal, xLocal, static_cast<T>(0), currentNum);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t startOffset = GetBlockIdx() * blockLength_;
    
    // 核心拦截：如果当前核的起始偏移超出总数据量，直接退出，防止非法内存访问
    if (startOffset >= totalNum_) {
        return;
    }

    // 计算当前核实际需要处理的数据长度（处理尾核）
    int64_t actualBlockLength = blockLength_;
    if (startOffset + actualBlockLength > totalNum_) {
        actualBlockLength = totalNum_ - startOffset;
    }
    if (actualBlockLength <= 0) {
        return;
    }

    // 通过循环处理，避免单次处理过大导致 UB 溢出
    int64_t loopCount = (actualBlockLength + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (actualBlockLength - i * ubLength_ < ubLength_) ?
                              (actualBlockLength - i * ubLength_) : ubLength_;
        if (currentNum <= 0) continue;
        
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H