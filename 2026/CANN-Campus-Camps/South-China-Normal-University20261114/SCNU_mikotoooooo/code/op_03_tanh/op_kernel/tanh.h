
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
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// 初始化
template <typename T>
__aicore__ inline void Tanh<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const TanhTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;

    int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx()) * blockFactor;

    blockLength_ = totalNum - blockOffset;

    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }

    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    ubLength_ = tilingData->ubFactor;

    // 设置输入输出 GlobalTensor
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset,
        blockLength_);

    // 初始化输入输出队列
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));
}

// 数据搬入
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        xLocal,
        inputGMX[progress],
        currentNum);

    inputQueueX.EnQue(xLocal);
}

// Tanh 计算
template <typename T>
__aicore__ inline void Tanh<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    // 调用 Ascend C Tanh 高阶 API
    // 由 API 框架管理内部临时空间
    AscendC::Tanh(
        yLocal,
        xLocal,
        static_cast<uint32_t>(currentNum));

    outputQueueY.EnQue(yLocal);

    inputQueueX.FreeTensor(xLocal);
}

// 数据搬出
template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress],
        yLocal,
        currentNum);

    outputQueueY.FreeTensor(yLocal);
}

// 执行计算
template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t progress = 0;

    while (progress < blockLength_) {
        int64_t currentNum = blockLength_ - progress;

        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);

        progress += currentNum;
    }
}

} // namespace NsTanh

#endif // TANH_H
