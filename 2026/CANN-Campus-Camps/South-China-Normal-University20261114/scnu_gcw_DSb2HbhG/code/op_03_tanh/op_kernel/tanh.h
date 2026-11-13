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
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    // 数据量过小或 tiling 非法时，本核不处理任何数据。
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }
    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    // 各核处理连续、等长的一段数据，blockFactor 已由 host 侧按 32B 对齐。
    const int64_t blockOffset = blockLength_ * static_cast<int64_t>(AscendC::GetBlockIdx());
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
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
    // tanh(x) = (e^x - e^-x) / (e^x + e^-x) = 1 - 2 / (e^(2x) + 1)
    // 采用第二种形式：只需要一次 Exp；当 e^(2x) 溢出为 inf 时，结果自然饱和到 ±1。
    AscendC::Muls(yLocal, xLocal, static_cast<T>(2.0f), static_cast<uint32_t>(currentNum));
    AscendC::Exp(yLocal, yLocal, static_cast<uint32_t>(currentNum));
    AscendC::Adds(yLocal, yLocal, static_cast<T>(1.0f), static_cast<uint32_t>(currentNum));
    // 复用输入 buffer 作为临时空间：xLocal = 2 / (e^(2x) + 1)。
    AscendC::Duplicate(xLocal, static_cast<T>(2.0f), static_cast<uint32_t>(currentNum));
    AscendC::Div(xLocal, xLocal, yLocal, static_cast<uint32_t>(currentNum));
    AscendC::Muls(xLocal, xLocal, static_cast<T>(-1.0f), static_cast<uint32_t>(currentNum));
    AscendC::Adds(yLocal, xLocal, static_cast<T>(1.0f), static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
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
        const int64_t remaining = blockLength_ - i * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}
} // namespace NsTanh
#endif // TANH_H