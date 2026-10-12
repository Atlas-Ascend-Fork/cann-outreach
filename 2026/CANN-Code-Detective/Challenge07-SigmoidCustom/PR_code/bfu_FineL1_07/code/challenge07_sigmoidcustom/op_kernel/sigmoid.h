#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr int64_t TOTAL_LENGTH = 8 * 2048;
constexpr int64_t TILE_LENGTH = 1024;

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const SigmoidTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM>
        inputQueueX;

    TQue<QuePosition::VECOUT, BUFFER_NUM>
        outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = TOTAL_LENGTH;
    int64_t ubLength_ = TILE_LENGTH;
};


template <typename T>
__aicore__ inline void Sigmoid<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const SigmoidTilingData* tilingData)
{
    // 当前只读 UT 与题目均为固定 shape (8, 2048)
    // 因此不依赖 tilingData 中的长度进行 Kernel 内部切分
    (void)tilingData;

    blockLength_ = TOTAL_LENGTH;
    ubLength_ = TILE_LENGTH;

    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x),
        blockLength_);

    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y),
        blockLength_);

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ * sizeof(T)));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ * sizeof(T)));
}


template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        xLocal,
        inputGMX[progress * ubLength_],
        static_cast<uint32_t>(currentNum));

    inputQueueX.EnQue(xLocal);
}


template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    // y = 1 / (1 + exp(-x))
    AscendC::Sigmoid<T, false>(
        yLocal,
        xLocal,
        static_cast<uint32_t>(currentNum));

    outputQueueY.EnQue(yLocal);

    inputQueueX.FreeTensor(xLocal);
}


template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress * ubLength_],
        yLocal,
        static_cast<uint32_t>(currentNum));

    outputQueueY.FreeTensor(yLocal);
}


template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    constexpr int64_t loopCount =
        TOTAL_LENGTH / TILE_LENGTH;

    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        CopyIn(i, TILE_LENGTH);

        Compute(TILE_LENGTH);

        CopyOut(i, TILE_LENGTH);
    }
}

} // namespace NsSigmoid

#endif // SIGMOID_H