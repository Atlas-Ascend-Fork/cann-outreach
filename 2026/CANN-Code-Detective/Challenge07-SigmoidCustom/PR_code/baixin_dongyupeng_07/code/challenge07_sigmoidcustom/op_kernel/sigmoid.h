/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

#include <type_traits>
// 注意：aicore 设备编译（bisheng）下 libstdc++ 的 std::min/std::max 为 host 函数，
//       不可在 kernel 内调用，标量取小/取大请用三元表达式。

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

/*!
 * \brief Sigmoid 逐元素算子 Kernel 类
 *
 * 计算链（DESIGN §1.4 / §2.4）：
 *   mode1（T=float）：Muls(-1) -> Exp -> Adds(1) -> Div(ones, den)，输入缓冲 in-place
 *   mode0（T=half） ：Cast(half->float) -> FP32 计算链 -> Cast(float->half)，规避 FP16 exp 溢出
 *
 * 缓冲规划（DESIGN §1.5）：
 *   inputQueueX(2) / outputQueueY(2)：双缓冲 VECIN/VECOUT
 *   onesBuf（FP32）：Div 被除数常数 1，Init 时一次填充
 *   f32Work（FP32，仅 mode0）：半精度升精度计算缓冲
 */
template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> onesBuf;  // FP32 常数 1（Div 被除数）
    TBuf<TPosition::VECCALC> f32Work;  // FP32 中间计算缓冲（仅 mode0 / T=half 使用）

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockFactor_ = 0;  // 每个核的元素数（Process 内计算 GM 偏移用）
    int64_t blockLength_ = 0;  // 本核实际元素数
    int64_t ubLength_ = 0;     // 单次 UB 循环元素数
};

template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    inputGMX.SetGlobalBuffer((__gm__ T*)x, tilingData->totalNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, tilingData->totalNum);

    blockFactor_ = tilingData->blockFactor;
    int64_t blockRemain = tilingData->totalNum - (int64_t)GetBlockIdx() * blockFactor_;
    blockLength_ = (blockFactor_ < blockRemain) ? blockFactor_ : blockRemain;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(onesBuf, ubLength_ * sizeof(float));
    Duplicate(onesBuf.Get<float>(), 1.0f, (int32_t)ubLength_);

    if constexpr (std::is_same_v<T, half>) {
        pipe.InitBuffer(f32Work, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inLocal = inputQueueX.AllocTensor<T>();
    int64_t off = (int64_t)GetBlockIdx() * blockFactor_ + progress * ubLength_;
    if ((currentNum * (int64_t)sizeof(T)) % 32 == 0) {
        // 整 tile：严格 32B 对齐，DataCopy
        DataCopy(inLocal, inputGMX[off], (uint32_t)currentNum);
    } else {
        // 尾 tile：非对齐，DataCopyPad（GM 侧 1B 对齐即可，UB 侧自动补 dummy）
        DataCopyPad(inLocal, inputGMX[off],
                    DataCopyExtParams(1, (uint32_t)(currentNum * (int64_t)sizeof(T)), 0, 0, 0),
                    DataCopyPadExtParams<T>(false, 0, 0, (T)0));
    }
    inputQueueX.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();

    if constexpr (std::is_same_v<T, float>) {
        // mode1：FP32 原 dtype 直算，输入缓冲 in-place
        Muls(xLocal, xLocal, (T)-1, (int32_t)currentNum);
        Exp(xLocal, xLocal, (int32_t)currentNum);
        Adds(xLocal, xLocal, (T)1, (int32_t)currentNum);
        Div(yLocal, onesLocal, xLocal, (int32_t)currentNum);  // y = 1/(1+exp(-x))
    } else {
        // mode0：half -> FP32 计算 -> half（升精度，规避 FP16 exp(-x) 溢出）
        LocalTensor<float> f32Local = f32Work.Get<float>();
        Cast<float, half>(f32Local, xLocal, RoundMode::CAST_NONE, (uint32_t)currentNum);
        Muls(f32Local, f32Local, -1.0f, (int32_t)currentNum);
        Exp(f32Local, f32Local, (int32_t)currentNum);
        Adds(f32Local, f32Local, 1.0f, (int32_t)currentNum);
        Div(f32Local, onesLocal, f32Local, (int32_t)currentNum);  // dst==src1 100% 重叠（允许）
        Cast<half, float>(yLocal, f32Local, RoundMode::CAST_RINT, (uint32_t)currentNum);
    }

    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t off = (int64_t)GetBlockIdx() * blockFactor_ + progress * ubLength_;
    if ((currentNum * (int64_t)sizeof(T)) % 32 == 0) {
        DataCopy(outputGMY[off], yLocal, (uint32_t)currentNum);
    } else {
        DataCopyPad(outputGMY[off], yLocal,
                    DataCopyExtParams(1, (uint32_t)(currentNum * (int64_t)sizeof(T)), 0, 0, 0));
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    // 本核按 blockFactor 步进，块内按 ubLength_ 循环；尾 tile 元素数可能小于 ubLength_
    int64_t loopN = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopN; i++) {
        int64_t blockRemain = blockLength_ - i * ubLength_;
        int64_t cur = (ubLength_ < blockRemain) ? ubLength_ : blockRemain;
        CopyIn(i, cur);
        Compute(cur);
        CopyOut(i, cur);
    }
}

} // namespace NsSigmoid
#endif // SIGMOID_H
