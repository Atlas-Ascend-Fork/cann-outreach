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
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t localLength_ = 0;
};

// 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (blockLength_ <= 0) {
        blockLength_ = 1;
    }
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t start = blockIdx * blockLength_;
    // 最后一个核处理剩余元素，兼容 totalNum 不能被 blockDim 整除的场景
    localLength_ = blockLength_;
    if (blockIdx == blockNum - 1) {
        localLength_ = totalNum_ - start;
    }
    if (localLength_ < 0) {
        localLength_ = 0;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + start, localLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + start, localLength_);

    // 双缓冲：按 ubFactor 分配输入/输出队列
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    // Tanh 高阶 API 的临时空间：float 需要 1 份，half 需要 2 份 float 空间
    pipe.InitBuffer(tmpBuffer, ubLength_ * sizeof(float) * 2);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<uint8_t> tmpLocal = tmpBuffer.Get<uint8_t>();
    // y = tanh(x)，逐元素计算
    AscendC::Tanh(yLocal, xLocal, tmpLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (localLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (localLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        // 最后一次 UB 循环处理不足 ubFactor 的尾块
        int64_t currentNum = ubLength_;
        if (i == loopCount - 1) {
            currentNum = localLength_ - i * ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
