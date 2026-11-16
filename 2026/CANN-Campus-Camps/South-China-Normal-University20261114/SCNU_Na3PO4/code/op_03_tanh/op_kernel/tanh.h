/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 *
 * tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * 仅使用 Exp / Muls / Sub / Add / Div / Cast 等基础矢量算子拼接实现。
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

// tanh 在 |x| >= 15 时 float32 已饱和为 ±1，clamp 不会损失精度，但能避免 exp 溢出
constexpr float TANH_CLAMP = 15.0f;

// 计算精度：float16 升到 float32 计算，避免 e^x - e^-x 在 0 附近的灾难性抵消
template <typename T>
struct TanhComputeTraits {
    using ComputeT = T;
};

template <>
struct TanhComputeTraits<half> {
    using ComputeT = float;
};

// 类型转换辅助：同类型时用 Adds(+0) 搬运，避免非法的 Cast 调用
template <typename SRC, typename DST>
struct TanhCast {
    __aicore__ inline static void Run(const LocalTensor<DST>& dst, const LocalTensor<SRC>& src, uint32_t len)
    {
        AscendC::Adds(dst, src, static_cast<DST>(0), len);
    }
};

template <>
struct TanhCast<half, float> {
    __aicore__ inline static void Run(const LocalTensor<float>& dst, const LocalTensor<half>& src, uint32_t len)
    {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, len);
    }
};

template <>
struct TanhCast<float, half> {
    __aicore__ inline static void Run(const LocalTensor<half>& dst, const LocalTensor<float>& src, uint32_t len)
    {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, len);
    }
};

template <typename T>
class Tanh {
public:
    using ComputeT = typename TanhComputeTraits<T>::ComputeT;

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
    // 计算临时变量（统一为 float32：fp16 输入时升精度，fp32 输入时同精度）
    TBuf<TPosition::VECCALC> calcBufX;
    TBuf<TPosition::VECCALC> calcBufExp;
    TBuf<TPosition::VECCALC> calcBufNegExp;
    TBuf<TPosition::VECCALC> calcBufTmp;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // 每个核处理的元素数（尾核可能不足 blockFactor）
    int64_t remainderLength =
        tilingData->totalNum - tilingData->blockFactor * static_cast<int64_t>(AscendC::GetBlockIdx());
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(calcBufX, ubLength_ * sizeof(ComputeT));
    pipe.InitBuffer(calcBufExp, ubLength_ * sizeof(ComputeT));
    pipe.InitBuffer(calcBufNegExp, ubLength_ * sizeof(ComputeT));
    pipe.InitBuffer(calcBufTmp, ubLength_ * sizeof(ComputeT));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    uint32_t len = static_cast<uint32_t>(currentNum);

    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    LocalTensor<ComputeT> x = calcBufX.Get<ComputeT>();
    LocalTensor<ComputeT> expX = calcBufExp.Get<ComputeT>();
    LocalTensor<ComputeT> expNegX = calcBufNegExp.Get<ComputeT>();
    LocalTensor<ComputeT> tmp = calcBufTmp.Get<ComputeT>();

    // 1. 统一到计算精度（fp16 -> fp32；fp32 为一次搬运）
    TanhCast<T, ComputeT>::Run(x, xLocal, len);

    // 2. clamp 到 [-15, 15]，避免 exp 溢出（区间外 tanh 已饱和）
    Mins(x, x, static_cast<ComputeT>(TANH_CLAMP), len);
    Maxs(x, x, static_cast<ComputeT>(-TANH_CLAMP), len);

    // 3. exp(x)、exp(-x)
    Exp(expX, x, len);
    Muls(tmp, x, static_cast<ComputeT>(-1.0f), len);
    Exp(expNegX, tmp, len);

    // 4. (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Sub(tmp, expX, expNegX, len);  // 分子
    Add(x, expX, expNegX, len);    // 分母（复用 x 的 buffer）
    Div(expNegX, tmp, x, len);     // tanh

    // 5. 转回输出类型
    TanhCast<ComputeT, T>::Run(yLocal, expNegX, len);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
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
