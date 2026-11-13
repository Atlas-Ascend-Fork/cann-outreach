#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {
using namespace AscendC;

/* 每核一轮就能搬完（blockFactor=2048 元素），单缓冲最省一次流水同步 */
constexpr int32_t BUFFER_NUM = 1;

__aicore__ inline int64_t CeilDivI64(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
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
    /* 计算过程需要的临时变量（都放 UB，用 float 存中间结果） */
    TBuf<TPosition::VECCALC> tmpBufExp;    // 存 2x、exp(2x)、中间结果
    TBuf<TPosition::VECCALC> tmpBufConst;  // 存常数 2
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

    /* 本核的起始偏移 = 核号 * 每核处理量 */
    int64_t offset = static_cast<int64_t>(GetBlockIdx()) * blockLength_;
    if (offset + blockLength_ > tilingData->totalNum) {
        blockLength_ = tilingData->totalNum - offset;
    }
    if (blockLength_ <= 0) {
        blockLength_ = 0;
        return;   /* 空核直接退出，不申请 UB */
    }
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBufExp, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBufConst, ubLength_ * sizeof(float));
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
    LocalTensor<float> expLocal = tmpBufExp.Get<float>();
    LocalTensor<float> constLocal = tmpBufConst.Get<float>();
    uint32_t num = static_cast<uint32_t>(currentNum);

    /* 统一转到 float 上算，float16 也能保证精度 */
    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(expLocal, xLocal, RoundMode::CAST_NONE, num);
    } else {
        DataCopy(expLocal, xLocal, currentNum);
    }

    /* tanh(x) = 1 - 2 / (exp(2x) + 1) */
    Muls(expLocal, expLocal, 2.0f, num);      /* 2x */
    Exp(expLocal, expLocal, num);             /* exp(2x) */
    Adds(expLocal, expLocal, 1.0f, num);      /* exp(2x) + 1 */
    Duplicate(constLocal, 2.0f, num);         /* 常数 2 */
    Div(expLocal, constLocal, expLocal, num); /* 2 / (exp(2x)+1) */
    Muls(expLocal, expLocal, -1.0f, num);     /* 取负 */
    Adds(expLocal, expLocal, 1.0f, num);      /* 1 - 2/(exp(2x)+1) = tanh(x) */

    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(yLocal, expLocal, RoundMode::CAST_RINT, num);
    } else {
        DataCopy(yLocal, expLocal, currentNum);
    }

    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue<T>(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t ubLoopCount = CeilDivI64(blockLength_, ubLength_);
    for (int64_t i = 0; i < ubLoopCount; ++i) {
        int64_t currentNum = (i == ubLoopCount - 1) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H