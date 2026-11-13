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
    // Tanh 计算所需的 fp32 中间结果（exp(2x)、exp(2x)±1）
    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// Init：绑定 GM 地址并按 tiling 参数分配 UB
//   blockLength_：本核实际处理元素数，尾核取剩余部分避免越界
//   ubLength_：   单次 UB 循环处理的元素数（已按 32B 对齐）
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t coreOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalNum - coreOffset;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + coreOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + coreOffset, blockLength_);

    // 输入/输出各双缓冲，另外两块 fp32 临时 buffer 存放 tanh 中间结果
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf0, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    // DataCopyPad 支持非 32B 对齐长度，尾块不会越界
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

// Compute：tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
//   fp16 输入先升到 fp32 计算，最后按 round-to-nearest 转回 fp16，避免半精度误差放大
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
    LocalTensor<float> tmp1 = tmpBuf1.Get<float>();

    // 矢量指令按 32B 对齐处理，尾部多余元素留在 buffer 内，CopyOut 只回搬有效部分
    int64_t alignElem = 32 / sizeof(T);
    int64_t calcCount = ((currentNum + alignElem - 1) / alignElem) * alignElem;

    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(tmp0, xLocal, 2.0f, calcCount); // 2x
    } else {
        Cast(tmp0, xLocal, RoundMode::CAST_NONE, calcCount); // fp16 -> fp32
        Muls(tmp0, tmp0, 2.0f, calcCount);                   // 2x
    }
    Exp(tmp1, tmp0, calcCount);          // e = exp(2x)
    Adds(tmp0, tmp1, 1.0f, calcCount);   // e + 1
    Adds(tmp1, tmp1, -1.0f, calcCount);  // e - 1

    if constexpr (sizeof(T) == sizeof(float)) {
        Div(yLocal, tmp1, tmp0, calcCount); // (e - 1) / (e + 1)
    } else {
        Div(tmp1, tmp1, tmp0, calcCount);
        Cast(yLocal, tmp1, RoundMode::CAST_RINT, calcCount); // fp32 -> fp16
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
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
