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

// kernel 侧不依赖 <type_traits>，自行提供一个类型相同的判断工具
template <typename T1, typename T2>
struct TanhIsSameType {
    static constexpr bool VALUE = false;
};
template <typename T1>
struct TanhIsSameType<T1, T1> {
    static constexpr bool VALUE = true;
};


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
    __aicore__ inline void CalcTanh(const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmp, int32_t calCount);
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    TBuf<QuePosition::VECCALC> calcBuf0; // fp32 临时变量：fp16/bf16 输入转 fp32、最终结果
    TBuf<QuePosition::VECCALC> calcBuf1; // fp32 临时变量：中间结果 1 + exp(2x)

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 单核总数据量与 UB 单次循环数据量
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    // 当前核在 GM 上的起始偏移，尾核需要截断实际处理的数据量
    int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    int64_t remainder = tilingData->totalNum - blockOffset;
    if (remainder < 0) {
        remainder = 0;
    }
    if (blockLength_ > remainder) {
        blockLength_ = remainder;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset, blockLength_);

    // 输入、输出各开双缓冲；另外两块 fp32 中间变量用于承载计算过程
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(calcBuf0, ubLength_ * sizeof(float));
    pipe.InitBuffer(calcBuf1, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    // 使用 DataCopyPad 兼容尾块非 32B 对齐的搬运，避免 GM 越界读
    DataCopyExtParams copyParams(1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0);
    DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CalcTanh(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                         const LocalTensor<float>& tmp, int32_t calCount)
{
    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) = 1 - 2 / (1 + exp(2x))
    // 采用后一种等价形式：只需一次 Exp，且 x 趋于 ±inf 时 2/(1+exp(2x)) 趋于 0/2，结果稳定收敛到 ±1，不会出现 inf/inf 的 NaN。
    Muls(tmp, src, 2.0f, calCount);      
    Exp(tmp, tmp, calCount);             
    Adds(tmp, tmp, 1.0f, calCount);      
    Duplicate(dst, 2.0f, calCount);      
    Div(dst, dst, tmp, calCount);       
    Muls(dst, dst, -1.0f, calCount);     
    Adds(dst, dst, 1.0f, calCount);      
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> tmpLocal = calcBuf1.Get<float>();

    const int32_t calCount = static_cast<int32_t>(currentNum);

    if constexpr (TanhIsSameType<T, float>::VALUE) {
        // fp32：直接在输入、输出 buffer 上计算，不需要 Cast
        CalcTanh(yLocal, xLocal, tmpLocal, calCount);
    } else {
        // fp16/bf16：先升精度到 fp32 计算，结果再转回原类型
        LocalTensor<float> castLocal = calcBuf0.Get<float>();
        Cast(castLocal, xLocal, RoundMode::CAST_NONE, calCount);  // castLocal = (float)x

        // src 只在 CalcTanh 首条 Muls 中被读取，之后 dst 与 src 复用同一块 buffer 是安全的
        CalcTanh(castLocal, castLocal, tmpLocal, calCount);       // castLocal = tanh(x)
        Cast(yLocal, castLocal, RoundMode::CAST_RINT, calCount);  // y = (T)tanh(x)
    }

    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    DataCopyExtParams copyParams(1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0);
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // TODO: 实现 Process 逻辑
    // 该核没有分配到数据，直接返回
    if (blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = blockLength_ - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
