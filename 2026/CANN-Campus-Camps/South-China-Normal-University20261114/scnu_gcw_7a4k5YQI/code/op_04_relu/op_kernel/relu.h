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
 
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};
 
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y,
                                     const ReluTilingData* tilingData)
{
    // 完全信任 host 传的 tiling
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t perCore = tilingData->blockFactor;   // host 算好的每核长度
    int64_t totalNum = tilingData->totalNum;
 
    int64_t startOffset = blockIdx * perCore;
    if (startOffset >= totalNum) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }
    blockLength_ = totalNum - startOffset;
    if (blockLength_ > perCore) {
        blockLength_ = perCore;
    }
 
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = blockLength_;
    }
    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
 
    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);
 
    pipe.InitBuffer(inputQueueX, BUFFER_NUM,
                    ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM,
                    ubLength_ * static_cast<int64_t>(sizeof(T)));
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
 
    // y = max(x, 0)
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), currentNum);
 
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
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = blockLength_ / ubLength_;
    int64_t remainder = blockLength_ % ubLength_;
 
    for (int64_t i = 0; i < loopCount; ++i) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }
    if (remainder > 0) {
        CopyIn(loopCount, remainder);
        Compute(remainder);
        CopyOut(loopCount, remainder);
    }
}
 
} // namespace NsRelu
#endif // RELU_H
