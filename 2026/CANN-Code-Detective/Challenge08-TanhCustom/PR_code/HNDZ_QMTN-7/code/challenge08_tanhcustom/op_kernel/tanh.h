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

    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // ★ 本核起点 = 核号 × 每核元素数（漏了这句，所有核都从头处理同一段 ⇒ 平台 97.5% 错）
    int64_t blockIdx = AscendC::GetBlockIdx();
    blockLength_ = tilingData->blockFactor;
    int64_t remain = tilingData->totalNum - blockIdx * tilingData->blockFactor;
    if (remain < blockLength_) {
        blockLength_ = (remain > 0) ? remain : 0;
    }
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockIdx * tilingData->blockFactor,
                             static_cast<uint32_t>(blockLength_));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockIdx * tilingData->blockFactor,
                              static_cast<uint32_t>(blockLength_));

    ubLength_ = tilingData->ubFactor / BUFFER_NUM;

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(tmpBuf1, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(tmpBuf2, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    if (currentNum * static_cast<int64_t>(sizeof(T)) % 32 == 0) {
        DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> tmp1 = tmpBuf1.Get<T>();
    LocalTensor<T> tmp2 = tmpBuf2.Get<T>();

    // ⚠️ 限幅常数必须用 float 存：bisheng 禁止在 aicore 函数里对 half 做算术运算
    const float clampF = (sizeof(T) == 4) ? 15.0f : 8.0f;

    Mins(tmp1, xLocal, static_cast<T>(clampF), currentNum);     // tmp1 = clamp(x)
    Maxs(tmp1, tmp1, static_cast<T>(-clampF), currentNum);
    Exp(tmp2, tmp1, currentNum);                                // tmp2 = a = e^x
    Muls(tmp1, tmp1, static_cast<T>(-1.0f), currentNum);        // tmp1 = -x
    Exp(yLocal, tmp1, currentNum);                              // yLocal = b = e^-x
    Sub(tmp1, tmp2, yLocal, currentNum);                        // tmp1 = d = a - b
    Add(tmp2, tmp2, yLocal, currentNum);                        // tmp2 = s = a + b
    Div(yLocal, tmp1, tmp2, currentNum);                        // yLocal = d / s

    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue<T>(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    if (currentNum * static_cast<int64_t>(sizeof(T)) % 32 == 0) {
        DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H