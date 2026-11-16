// /*!
//  * \file relu.h
//  * \brief Relu 算子 kernel 类定义
//  */

// #ifndef RELU_H
// #define RELU_H

// #include "kernel_operator.h"
// #include "kernel_tiling/kernel_tiling.h"
// #include "relu_tiling_data.h"
// #include "relu_tiling_key.h"

// namespace NsRelu {

// using namespace AscendC;

// constexpr int32_t BUFFER_NUM = 2;

// template <typename T>
// class Relu {
// public:
//     __aicore__ inline Relu(){};

//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
//     __aicore__ inline void Process();

// private:
//     __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
//     __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
//     __aicore__ inline void Compute(int64_t currentNum);

// private:
//     TPipe pipe;
//     TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
//     TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

//     GlobalTensor<T> inputGMX;
//     GlobalTensor<T> outputGMY;

//     int64_t blockLength_ = 0;
//     int64_t ubLength_ = 0;
// };

// // TODO: 实现具体的 kernel 逻辑
// template <typename T>
// __aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
// {
//     // TODO: 实现 Init 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
// {
//     // TODO: 实现 CopyIn 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::Compute(int64_t currentNum)
// {
//     // TODO: 实现 Compute 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
// {
//     // TODO: 实现 CopyOut 逻辑
// }

// template <typename T>
// __aicore__ inline void Relu<T>::Process()
// {
//     // TODO: 实现 Process 逻辑
// }

// } // namespace NsRelu
// #endif // RELU_H




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
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t currentNum);

private:
    TPipe pipe;
    // 使用官方推荐的 TQue 队列，精确管理数据流
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalLength_ = 0;
    int64_t coreOffset_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    totalLength_ = tilingData->totalNum;

    // 计算当前核的全局偏移量（严格按 blockLength_ 偏移）
    coreOffset_ = blockLength_ * AscendC::GetBlockIdx();
    
    // 这里直接绑定全局指针，不带偏移，偏移在 Process 里算，更安全
    inputGMX.SetGlobalBuffer((__gm__ T*)x);
    outputGMY.SetGlobalBuffer((__gm__ T*)y);

    // 严格按需分配，不加多余的 32 字节
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // 搬运对齐的数据
    DataCopy(xLocal, inputGMX[gmOffset], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // ReLU 计算：y = max(x, 0)
    Maxs(yLocal, xLocal, static_cast<T>(0), currentNum);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    // 搬运到对应的 GM 地址
    DataCopy(outputGMY[gmOffset], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t currentBlockLength = blockLength_;
    
    // 边界截断：如果该核分到的数据超出了总长度，则截断
    if (coreOffset_ + currentBlockLength > totalLength_) {
        currentBlockLength = totalLength_ - coreOffset_;
    }
    if (currentBlockLength <= 0) {
        return;
    }

    int64_t loopCount = (currentBlockLength + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (currentBlockLength - i * ubLength_) : ubLength_;
        if (currentNum <= 0) {
            break;
        }
        
        // 计算准确的 GM 偏移
        int64_t offset = coreOffset_ + i * ubLength_;
        
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H