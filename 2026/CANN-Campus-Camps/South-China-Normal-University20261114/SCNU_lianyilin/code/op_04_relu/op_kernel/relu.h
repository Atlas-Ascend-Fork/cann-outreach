/*!
 * \file relu.h
 * \brief 匹配 CANNJudge 2980 空工程的双缓冲 Relu
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
    __aicore__ inline Relu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

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
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    if (tilingData->totalNum <= 0 || tilingData->blockFactor <= 0 ||
        tilingData->ubFactor <= 0) {
        return;
    }
    const int64_t offset =
        static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum) {
        return;
    }
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;

    // 兼容模板 CPU UT 中 ubFactor=16384 的 float 用例，限制单块 UB 用量。
    const int64_t maxTileElements = (32 * 1024) / sizeof(T);
    const int64_t alignElements = 32 / sizeof(T);
    ubLength_ = tilingData->ubFactor < maxTileElements ? tilingData->ubFactor : maxTileElements;
    ubLength_ = (ubLength_ + alignElements - 1) / alignElements * alignElements;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    if (currentNum % (32 / sizeof(T)) == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        // 非对齐尾块按真实字节数搬入，避免读取输入之外的数据。
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T>(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    if (currentNum % (32 / sizeof(T)) == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad<T>(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    const int64_t firstNum = blockLength_ < ubLength_ ? blockLength_ : ubLength_;
    CopyIn(0, firstNum);
    for (int64_t i = 0; i < loopCount; ++i) {
        // 预取下一块，同时保证输入队列中最多只有两个块。
        if (i + 1 < loopCount) {
            const int64_t nextRemaining = blockLength_ - (i + 1) * ubLength_;
            const int64_t nextNum = nextRemaining < ubLength_ ? nextRemaining : ubLength_;
            CopyIn(i + 1, nextNum);
        }
        const int64_t remaining = blockLength_ - i * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

}  // namespace NsRelu
#endif  // RELU_H
