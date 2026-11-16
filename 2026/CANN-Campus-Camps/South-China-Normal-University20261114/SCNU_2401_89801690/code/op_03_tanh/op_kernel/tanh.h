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

// 输入绝对值上限：|x| > 20 时 tanh 在 fp32/fp16 下已经饱和为 ±1，
// 先做 clamp 可以保证 exp(2x) 不会溢出（fp32 下 exp(40) 仍在可表示范围内）
constexpr float MAX_ABS_X = 20.0f;

/*!
 * \brief 在 fp32 上计算 tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
 * \param dst 计算结果（fp32，需与 src/tmp 不同的 UB 块）
 * \param src 输入（fp32）
 * \param tmp 中间结果（fp32，可与 src 为同一块 UB）
 */
__aicore__ inline void TanhFp32(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                const LocalTensor<float>& tmp, uint32_t calCount)
{
    // 先把输入限制到 [-MAX_ABS_X, MAX_ABS_X]，避免 exp 上溢成 inf 后出现 inf/inf = nan
    Maxs(tmp, src, -MAX_ABS_X, calCount);
    Mins(tmp, tmp, MAX_ABS_X, calCount);
    Muls(tmp, tmp, 2.0f, calCount);   // 2x
    Exp(tmp, tmp, calCount);          // t = exp(2x)
    Adds(dst, tmp, -1.0f, calCount);  // t - 1
    Adds(tmp, tmp, 1.0f, calCount);   // t + 1
    Div(dst, dst, tmp, calCount);     // (t - 1) / (t + 1)
}

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
    // fp32 中间结果缓冲区：fp16 输入时也先升到 fp32 计算，避免半精度下 1 - 小量 的抵消误差
    TBuf<TPosition::VECCALC> calcBufA;
    TBuf<TPosition::VECCALC> calcBufB;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // 本核负责的元素个数
    int64_t ubLength_ = 0;     // 单次 UB 循环处理的元素个数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t startOffset = blockIdx * tilingData->blockFactor;
    const int64_t remainder = tilingData->totalNum - startOffset;
    blockLength_ = (remainder > tilingData->blockFactor) ? tilingData->blockFactor : remainder;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = (tilingData->ubFactor > 0) ? tilingData->ubFactor : 1;

    // 每个队列 BUFFER_NUM 块，每块 ubLength_ 个元素（host 侧已保证 32B 对齐）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(calcBufA, static_cast<uint32_t>(ubLength_ * sizeof(float)));
    pipe.InitBuffer(calcBufB, static_cast<uint32_t>(ubLength_ * sizeof(float)));

    if (blockLength_ == 0) {
        return;  // 本核没有分到数据，Process 中直接返回
    }

    // 本核负责 GM 上的 [startOffset, startOffset + blockLength_) 区间
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + startOffset, blockLength_);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // 使用 DataCopyPad：起始地址按 32B 对齐，尾块长度允许不是 32B 的整数倍
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    const uint32_t calCount = static_cast<uint32_t>(currentNum);
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> tmpLocal = calcBufA.Get<float>();

    if constexpr (sizeof(T) == sizeof(half)) {
        // fp16：x -> fp32，在 fp32 上计算 tanh，再一次性转回 fp16
        LocalTensor<float> dstLocal = calcBufB.Get<float>();
        Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, calCount);
        TanhFp32(dstLocal, tmpLocal, tmpLocal, calCount);
        Cast(yLocal, dstLocal, RoundMode::CAST_NONE, calCount);
    } else {
        TanhFp32(yLocal, xLocal, tmpLocal, calCount);
    }

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
    if (blockLength_ <= 0) {
        return;
    }
    // 单核内按 ubLength_ 分块，循环执行：搬入 -> 计算 -> 搬出
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H