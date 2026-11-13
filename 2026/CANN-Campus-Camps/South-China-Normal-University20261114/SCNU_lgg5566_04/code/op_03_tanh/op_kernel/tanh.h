/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 *
 * tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) = 1 - 2 / (exp(2x) + 1)
 *
 * 设计要点（不使用 Tanh 高阶 API，只用基础小算子拼接）：
 * 1. 多核：按 blockFactor 切分，blockFactor 已按 64 元素对齐，首地址天然 32B 对齐；
 * 2. 双缓冲：TQue BUFFER_NUM=2，DMA 与 Vector 计算流水重叠；
 * 3. fp16 输入统一 Cast 到 fp32 计算后一次 Cast 回 fp16，避免 fp16 Exp 溢出；
 * 4. |x| < 0.55 走三次多项式（CANN 官方 tanh 同款最小误差系数），保证近零精度；
 * 5. |x| >= 0.55 走 exp 公式，先 Mins(x, 20) 防止 exp 溢出；Select 分段。
 *
 * 说明：ubFactor 仅作为搬运粒度提示，kernel 内部再按 MAX_UB_ELEMS 夹紧，
 *       保证即使外部传入很大的 ubFactor，UB 占用仍有上界。
 */

#ifndef TANH_H
#define TANH_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;    // 每队列 buffer 块数（双缓冲）
constexpr int64_t ALIGN_ELEM = 64;   // 64 元素对齐（满足 32B 搬运 + 完整 vector repeat）
constexpr int64_t MAX_UB_ELEMS = 2048; // 单块元素数上限，控制 UB 占用

// 多项式分段阈值与系数（CANN 内置 tanh 实现所用最小误差系数）
constexpr float TANH_POLY_BOUND = 0.55f;
constexpr float TANH_EXP_CLAMP = 20.0f;
constexpr float TANH_COEF_3 = 0.0157396831f;
constexpr float TANH_COEF_2 = -0.0523039624f;
constexpr float TANH_COEF_1 = 0.133152977f;
constexpr float TANH_COEF_0 = -0.333327681f;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void TanhCompute(const LocalTensor<float>& x, LocalTensor<float>& y, int64_t count);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<TPosition::VECCALC> squareBuf;   // x^2
    TBuf<TPosition::VECCALC> polyBuf;     // 多项式结果
    TBuf<TPosition::VECCALC> tempBufA;    // |x| / 分子
    TBuf<TPosition::VECCALC> tempBufB;    // exp 中间量 / 分母
    TBuf<TPosition::VECCALC> maskBuf;     // 比较结果掩码
    TBuf<TPosition::VECCALC> castBuf;     // fp16 -> fp32
    TBuf<TPosition::VECCALC> resultBuf;   // 分段选择结果

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0; // 每核处理的元素数
    int64_t ubLength_ = 0;    // 单次搬运/计算的元素数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t start = blockIdx * tilingData->blockFactor;
    const int64_t remain = tilingData->totalNum - start;
    if (remain <= 0) {
        blockLength_ = 0;
    } else {
        blockLength_ = (remain > tilingData->blockFactor) ? tilingData->blockFactor : remain;
    }

    // ubFactor 先按 MAX_UB_ELEMS 夹紧，再按 64 元素对齐，保证搬运与 Compare 都对齐
    int64_t ub = tilingData->ubFactor;
    if (ub <= 0) {
        ub = tilingData->blockFactor;
    }
    if (ub > MAX_UB_ELEMS) {
        ub = MAX_UB_ELEMS;
    }
    ubLength_ = (ub + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
    if (ubLength_ <= 0) {
        ubLength_ = ALIGN_ELEM;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, (blockLength_ > 0) ? blockLength_ : 0);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, (blockLength_ > 0) ? blockLength_ : 0);

    // 输入/输出队列（双缓冲）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));

    // 计算临时变量（统一 fp32 计算）
    pipe.InitBuffer(squareBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(polyBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(tempBufA, ubLength_ * sizeof(float));
    pipe.InitBuffer(tempBufB, ubLength_ * sizeof(float));
    pipe.InitBuffer(maskBuf, (ubLength_ + 31) / 32 * 32);
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(castBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(resultBuf, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = blockLength_ - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    if constexpr (std::is_same<T, float>::value) {
        TanhCompute(xLocal, yLocal, currentNum);
    } else {
        LocalTensor<float> xFloat = castBuf.Get<float>();
        LocalTensor<float> yFloat = resultBuf.Get<float>();
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, currentNum);
        TanhCompute(xFloat, yFloat, currentNum);
        Cast(yLocal, yFloat, RoundMode::CAST_RINT, currentNum);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::TanhCompute(const LocalTensor<float>& x, LocalTensor<float>& y, int64_t count)
{
    LocalTensor<float> square = squareBuf.Get<float>();
    LocalTensor<float> poly = polyBuf.Get<float>();
    LocalTensor<float> tempA = tempBufA.Get<float>();
    LocalTensor<float> tempB = tempBufB.Get<float>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();

    // poly = x + x * (t * (c0 + t * (c1 + t * (c2 + t * c3)))), t = x * x
    Mul(square, x, x, count);
    Muls(poly, square, TANH_COEF_3, count);
    Adds(poly, poly, TANH_COEF_2, count);
    Mul(poly, poly, square, count);
    Adds(poly, poly, TANH_COEF_1, count);
    Mul(poly, poly, square, count);
    Adds(poly, poly, TANH_COEF_0, count);
    Mul(poly, poly, square, count);
    Mul(poly, poly, x, count);
    Add(poly, poly, x, count);

    // exp 公式：y = (exp(2x) - 1) / (exp(2x) + 1)，x 截断到 20 防溢出
    Abs(tempA, x, count);
    Compares(mask, tempA, TANH_POLY_BOUND, CMPMODE::LT, count);
    Mins(tempB, x, TANH_EXP_CLAMP, count);
    Muls(tempB, tempB, 2.0f, count);
    Exp(tempB, tempB, count);
    Adds(tempA, tempB, -1.0f, count);
    Adds(tempB, tempB, 1.0f, count);
    Div(tempA, tempA, tempB, count);

    // 小值区间用多项式，其余用 exp 公式
    Select(y, mask, poly, tempA, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

} // namespace NsTanh
#endif // TANH_H
