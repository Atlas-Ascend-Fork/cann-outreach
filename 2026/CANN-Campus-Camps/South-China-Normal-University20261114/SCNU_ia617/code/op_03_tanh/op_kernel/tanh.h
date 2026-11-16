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

    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> maskBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // 本核处理的元素数（尾核取剩余部分）
    int64_t ubLength_ = 0;     // UB 单次循环处理的元素数
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Init：绑定 GM 张量并初始化 UB 队列（Double Buffer）
//   blockLength_：totalNum 减去前面核已分配长度后与 blockFactor 取小，
//                 保证尾核不越界
//   ubLength_：   tiling 下发的单次 UB 循环元素数
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(workBuf, 5 * ubLength_ * sizeof(float));
    pipe.InitBuffer(maskBuf, ((ubLength_ + 255) / 256) * 32);
}

// CopyIn：把当前 tile 的 x 从 GM 搬入 UB 并入队
//   currentNum 为尾块实际元素数，DataCopyPad 可搬任意长度，避免尾块越界
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

// Compute：出队 x，逐元素计算 y = tanh(x)，结果入队
//   矢量指令要求计算长度按 32 字节对齐（float 为 8 个元素、half 为 16 个），
//   不足时向上对齐计算，多余部分仍在 buffer 范围内，CopyOut 只回拷有效部分
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> x = inputQueueX.DeQue<T>();
    LocalTensor<T> y = outputQueueY.AllocTensor<T>();
    const uint32_t n = static_cast<uint32_t>(ubLength_);
    LocalTensor<float> work = workBuf.Get<float>();
    LocalTensor<float> x32 = work;
    LocalTensor<float> absX = work[n];
    LocalTensor<float> expTerm = work[2 * n];
    LocalTensor<float> denominator = work[3 * n];
    LocalTensor<float> magnitude = work[4 * n];
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    if constexpr (sizeof(T) == sizeof(float)) {
        if (currentNum == n) {
            x32 = x;
        } else {
            AscendC::Duplicate(x32, 0.0f, n);
            AscendC::Muls(x32, x, 1.0f, static_cast<uint32_t>(currentNum));
        }
    } else {
        if (currentNum < n) {
            AscendC::Duplicate(x32, 0.0f, n);
        }
        AscendC::Cast(x32, x, RoundMode::CAST_NONE, static_cast<uint32_t>(currentNum));
    }

    // e = exp(-2|x|)，|tanh(x)| = (1-e)/(1+e)。指数为非正数。
    AscendC::Abs(absX, x32, n);
    AscendC::Muls(expTerm, absX, -2.0f, n);
    AscendC::Exp(expTerm, expTerm, n);
    AscendC::Adds(denominator, expTerm, 1.0f, n);
    AscendC::Muls(expTerm, expTerm, -1.0f, n);
    AscendC::Adds(expTerm, expTerm, 1.0f, n);
    AscendC::Div(magnitude, expTerm, denominator, n);

    // 恢复符号。接近零时 tanh(x) = x + O(x^3)，直接取 x 避免相消。
    AscendC::Muls(denominator, magnitude, -1.0f, n);
    AscendC::CompareScalar(mask, x32, 0.0f, CMPMODE::LT, n);
    AscendC::Select(expTerm, mask, denominator, magnitude, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    AscendC::CompareScalar(mask, absX, 0.001f, CMPMODE::LT, n);
    if constexpr (sizeof(T) == sizeof(float)) {
        AscendC::Select(y, mask, x32, expTerm, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    } else {
        AscendC::Select(magnitude, mask, x32, expTerm, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        AscendC::Cast(y, magnitude, RoundMode::CAST_RINT, n);
    }
    outputQueueY.EnQue<T>(y);
    inputQueueX.FreeTensor(x);
}

// CopyOut：出队结果，从 UB 回拷 GM（只搬有效部分 currentNum）
template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

// Process：按 ubLength_ 切块循环，CopyIn → Compute → CopyOut
//   最后一块用剩余元素数 currentNum，避免尾块越界
template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H