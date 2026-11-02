/*!
 * \file sqrt.h
 * \brief Sqrt 算子 kernel 实现
 */

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace NsSqrt {

using namespace AscendC;

// 本题是单次逐元素运算。每个核通常只处理一个 tile，因此使用单缓冲，
// 避免为本题固定的小张量分配不必要的双缓冲空间。
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Sqrt {
public:
    __aicore__ inline Sqrt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData);
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

    // 当前核负责的元素数，以及每次 UB 迭代处理的元素数。
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Sqrt<T>::Init(
    GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    const int64_t totalNum = tilingData->totalNum;
    const int64_t blockFactor = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    const int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx()) * blockFactor;

    // 兼容空张量或无有效工作量的 block。
    if (totalNum <= 0 || blockFactor <= 0 || ubLength_ <= 0 ||
        blockOffset >= totalNum) {
        blockLength_ = 0;
        return;
    }

    const int64_t remain = totalNum - blockOffset;
    blockLength_ = (remain < blockFactor) ? remain : blockFactor;

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset, blockLength_);

    // 输入、输出各使用一个队列缓冲区。对本题的 16384 个元素，
    // 单核工作量较小，单缓冲可避免无收益的双缓冲空间开销。
    const uint32_t bufferBytes =
        static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(
        xLocal,
        inputGMX[progress * ubLength_],
        static_cast<int32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 直接使用 Ascend C 向量 Sqrt，分别支持 float32 和 float16。
    // 对 float16 不额外转换到 float32，减少转换指令和临时数据搬运。
    AscendC::Sqrt(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(
        outputGMY[progress * ubLength_],
        yLocal,
        static_cast<int32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount =
        (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t offset = progress * ubLength_;
        const int64_t remain = blockLength_ - offset;
        const int64_t currentNum =
            (remain < ubLength_) ? remain : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSqrt
#endif // SQRT_H
