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
#include <type_traits>

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr float TANH_CLIP = 8.8f;

__aicore__ inline void TanhFormula(const LocalTensor<float>& out, const LocalTensor<float>& work,
    int32_t count)
{
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1), x clipped to [-8.8, 8.8] to avoid overflow.
    Mins(work, work, TANH_CLIP, count);
    Maxs(work, work, -TANH_CLIP, count);
    Muls(work, work, 2.0f, count);
    Exp(work, work, count);
    Adds(out, work, 1.0f, count);
    Adds(work, work, -1.0f, count);
    Div(out, work, out, count);
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
    TBuf<QuePosition::VECCALC> floatBuffer;
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
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum) { return; }
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    if constexpr (std::is_same<T, half>::value) {
        pipe.InitBuffer(floatBuffer, 2 * ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> local = inputQueueX.AllocTensor<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(local, inputGMX[progress], params, padding);
    inputQueueX.EnQue(local);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const int32_t count = static_cast<int32_t>(currentNum);
    if constexpr (std::is_same<T, half>::value) {
        LocalTensor<float> xFloat = floatBuffer.Get<float>();
        LocalTensor<float> yFloat = xFloat[ubLength_];
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, count);
        TanhFormula(yFloat, xFloat, count);
        Cast(yLocal, yFloat, RoundMode::CAST_NONE, count);
    } else {
        TanhFormula(yLocal, xLocal, count);
    }
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> local = outputQueueY.DeQue<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress], local, params);
    outputQueueY.FreeTensor(local);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }
}

} // namespace NsTanh
#endif // TANH_H
