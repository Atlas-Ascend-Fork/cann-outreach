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
constexpr uint32_t UB_BUDGET = 160 * 1024; // 单核 UB 预算（为临时变量留出余量）
constexpr uint32_t ALIGN_BYTES = 32;       // DataCopy 要求 32B 对齐

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
    __aicore__ inline void ComputeTanh(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                       int32_t calCount);
    __aicore__ inline void ComputeTanh(const LocalTensor<half>& dst, const LocalTensor<half>& src,
                                       int32_t calCount);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tempBuffer;  // float 计算的中间结果
    TBuf<TPosition::VECCALC> tempBuffer2; // float 计算的分母临时张量

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0; // 本核处理的元素数
    int64_t ubLength_ = 0;    // 单次搬运 / 计算的元素数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    constexpr uint32_t alignNum = ALIGN_BYTES / sizeof(T);
    const int64_t totalLength = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    // 按 UB 预算收缩单次处理量：输入/输出双缓冲 + 两个 float 临时张量，并保证 32B 对齐
    constexpr uint32_t perElementBytes = sizeof(T) * 2 * BUFFER_NUM + 2 * sizeof(float);
    uint32_t maxUbLength = UB_BUDGET / perElementBytes;
    maxUbLength = maxUbLength / alignNum * alignNum;
    if (maxUbLength < alignNum) {
        maxUbLength = alignNum;
    }
    if (ubLength_ > static_cast<int64_t>(maxUbLength)) {
        ubLength_ = static_cast<int64_t>(maxUbLength);
    }

    // 本核在 GM 上的起始偏移，末核长度不足时按实际长度截断
    int64_t offset = blockLength_ * static_cast<int64_t>(AscendC::GetBlockIdx());
    if (offset >= totalLength) {
        blockLength_ = 0;
        offset = totalLength;
    } else if (offset + blockLength_ > totalLength) {
        blockLength_ = totalLength - offset;
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    // 为队列分配 UB 内存
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    // float16 输入转 float 计算所用的中间张量
    pipe.InitBuffer(tempBuffer, ubLength_ * sizeof(float));
    pipe.InitBuffer(tempBuffer2, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    ComputeTanh(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)，使用 Div 保证精度
template <typename T>
__aicore__ inline void Tanh<T>::ComputeTanh(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                            int32_t calCount)
{
    LocalTensor<float> tmpLocal = tempBuffer2.Get<float>();
    Muls(dst, src, 2.0f, calCount);      // 2x
    Maxs(dst, dst, -30.0f, calCount);    // 限幅，避免 exp 溢出
    Mins(dst, dst, 30.0f, calCount);
    Exp(dst, dst, calCount);             // exp(2x)
    Adds(tmpLocal, dst, 1.0f, calCount); // exp(2x) + 1
    Adds(dst, dst, -1.0f, calCount);     // exp(2x) - 1
    Div(dst, dst, tmpLocal, calCount);   // (exp(2x) - 1) / (exp(2x) + 1)
}

// float16 场景：先转 float 计算，再转回 float16
template <typename T>
__aicore__ inline void Tanh<T>::ComputeTanh(const LocalTensor<half>& dst, const LocalTensor<half>& src,
                                            int32_t calCount)
{
    LocalTensor<float> tmpLocal = tempBuffer.Get<float>();
    Cast(tmpLocal, src, RoundMode::CAST_NONE, calCount);
    ComputeTanh(tmpLocal, tmpLocal, calCount);
    Cast(dst, tmpLocal, RoundMode::CAST_RINT, calCount);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
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
        int64_t currentNum = ubLength_;
        if (i == loopCount - 1) {
            currentNum = blockLength_ - i * ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
