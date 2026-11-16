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

    // 计算过程所需临时变量：exp(x)、exp(-x)、-x，以及 fp16 升 fp32 计算所需的缓存
    TQue<QuePosition::VECCALC, 1> expXQueue;
    TQue<QuePosition::VECCALC, 1> expNegXQueue;
    TQue<QuePosition::VECCALC, 1> negXQueue;
    TQue<QuePosition::VECCALC, 1> castQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// ---------------------------------------------------------------------------
// Init：多核切分 + UB 内存申请
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t offset = blockIdx * blockLength_;
    if (offset >= totalNum) {
        blockLength_ = 0;  // 该核没有数据可算
    } else if (offset + blockLength_ > totalNum) {
        blockLength_ = totalNum - offset;  // 最后一个核处理剩余尾块
    }

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;  // 空张量或核无数据，无需申请 UB
    }

    inputGMX.SetGlobalBuffer((__gm__ T *)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + offset, blockLength_);

    // 输入/输出各双 buffer
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    // 计算用 fp32 临时缓存（fp16 输入时全部在 fp32 上计算，保证精度）
    pipe.InitBuffer(expXQueue, 1, ubLength_ * sizeof(float));
    pipe.InitBuffer(expNegXQueue, 1, ubLength_ * sizeof(float));
    pipe.InitBuffer(negXQueue, 1, ubLength_ * sizeof(float));
    if constexpr (sizeof(T) != sizeof(float)) {
        pipe.InitBuffer(castQueue, 1, ubLength_ * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// CopyIn：GM -> UB，使用 DataCopyPad 兼容非对齐尾块
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

// ---------------------------------------------------------------------------
// Compute：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// fp16 输入先升精度到 fp32 计算，最后四舍五入回 fp16
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    LocalTensor<float> expX = expXQueue.AllocTensor<float>();        // exp(x)
    LocalTensor<float> expNegX = expNegXQueue.AllocTensor<float>();  // exp(-x)
    LocalTensor<float> negX = negXQueue.AllocTensor<float>();        // -x / 分子

    LocalTensor<float> xf;
    if constexpr (sizeof(T) == sizeof(float)) {
        xf = xLocal;  // fp32 输入直接使用，无需额外缓存
    } else {
        xf = castQueue.AllocTensor<float>();
        Cast(xf, xLocal, RoundMode::CAST_NONE, currentNum);  // fp16 -> fp32
    }

    Exp(expX, xf, currentNum);                                       // exp(x)
    Muls(negX, xf, static_cast<float>(-1), currentNum);              // -x
    Exp(expNegX, negX, currentNum);                                  // exp(-x)
    Sub(negX, expX, expNegX, currentNum);                            // 分子 = exp(x) - exp(-x)
    Add(expX, expX, expNegX, currentNum);                            // 分母 = exp(x) + exp(-x)

    if constexpr (sizeof(T) == sizeof(float)) {
        Div(yLocal, negX, expX, currentNum);                         // fp32 直接输出
    } else {
        Div(negX, negX, expX, currentNum);                           // fp32 上完成除法
        Cast(yLocal, negX, RoundMode::CAST_RINT, currentNum);        // fp32 -> fp16（就近舍入）
        castQueue.FreeTensor(xf);
    }

    expXQueue.FreeTensor(expX);
    expNegXQueue.FreeTensor(expNegX);
    negXQueue.FreeTensor(negX);

    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue<T>(yLocal);
}

// ---------------------------------------------------------------------------
// CopyOut：UB -> GM
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

// ---------------------------------------------------------------------------
// Process：按 ubLength_ 分块循环，末块长度单独处理
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t remain = blockLength_ - i * ubLength_;
        int64_t currentNum = remain > ubLength_ ? ubLength_ : remain;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
