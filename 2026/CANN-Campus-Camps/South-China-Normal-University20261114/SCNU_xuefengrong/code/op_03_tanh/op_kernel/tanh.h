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
// 向量计算的最小对齐粒度（Byte）
constexpr int64_t UB_BLOCK_BYTES = 32;

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
    // tanh 计算所需临时 buffer：2 块 float，分别存放 exp(2x) 与分母/分子
    TBuf<QuePosition::VECCALC> tmpBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // 当前核在全局数据上的起始偏移与需要处理的元素个数（尾核会少一些）
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t gmOffset = blockIdx * tilingData->blockFactor;
    const int64_t remainNum = tilingData->totalNum - gmOffset;
    blockLength_ = remainNum <= 0 ? 0
                                  : (remainNum < tilingData->blockFactor ? remainNum : tilingData->blockFactor);
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + gmOffset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + gmOffset, blockLength_);

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    // 输入/输出队列：双 buffer
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    // 临时 buffer：两块 float，每块可容纳一个 ubLength_ 的 T 元素（fp16 走 float 计算）
    pipe.InitBuffer(tmpBuf, static_cast<uint32_t>(ubLength_ * sizeof(float) * 2));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    // 允许尾块非 32B 对齐，统一用 pad 搬运
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    LocalTensor<float> tmpA = tmpBuf.Get<float>();
    LocalTensor<float> tmpB = tmpBuf.Get<float>()[ubLength_];

    // 尾块按 32B 元素数向上对齐后再计算，避免向量指令长度不对齐；
    // 多算出的尾部元素不会被 CopyOut 搬出，结果正确性不受影响。
    const int32_t calcNum = static_cast<int32_t>((currentNum + (UB_BLOCK_BYTES / sizeof(float)) - 1) /
                                                 (UB_BLOCK_BYTES / sizeof(float)) *
                                                 (UB_BLOCK_BYTES / sizeof(float)));

    // tanh(x) = 1 - 2 / (exp(2x) + 1)
    if constexpr (sizeof(T) == sizeof(float)) {
        // fp32：直接在输入/输出 buffer 上原地运算
        LocalTensor<float> xf = xLocal.template ReinterpretCast<float>();
        LocalTensor<float> yf = yLocal.template ReinterpretCast<float>();

        Muls(tmpA, xf, 2.0f, calcNum);        // tmpA = 2x
        Exp(tmpB, tmpA, calcNum);             // tmpB = exp(2x)
        Adds(tmpA, tmpB, 1.0f, calcNum);      // tmpA = exp(2x) + 1
        Duplicate(tmpB, -2.0f, calcNum);      // tmpB = -2
        Div(tmpB, tmpB, tmpA, calcNum);       // tmpB = -2 / (exp(2x) + 1)
        Adds(yf, tmpB, 1.0f, calcNum);        // y = 1 - 2 / (exp(2x) + 1)
    } else {
        // fp16/bf16：先升精度到 fp32 计算，再按就近舍入落回原精度
        Cast(tmpA, xLocal, RoundMode::CAST_NONE, calcNum);   // tmpA = (float)x
        Muls(tmpA, tmpA, 2.0f, calcNum);                    // tmpA = 2x
        Exp(tmpB, tmpA, calcNum);                           // tmpB = exp(2x)
        Adds(tmpA, tmpB, 1.0f, calcNum);                    // tmpA = exp(2x) + 1
        Duplicate(tmpB, -2.0f, calcNum);                    // tmpB = -2
        Div(tmpB, tmpB, tmpA, calcNum);                     // tmpB = -2 / (exp(2x) + 1)
        Adds(tmpA, tmpB, 1.0f, calcNum);                    // tmpA = 1 - 2 / (exp(2x) + 1)
        Cast(yLocal, tmpA, RoundMode::CAST_RINT, calcNum);   // 转回 fp16/bf16
    }

    outputQueueY.EnQue(yLocal);
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
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H