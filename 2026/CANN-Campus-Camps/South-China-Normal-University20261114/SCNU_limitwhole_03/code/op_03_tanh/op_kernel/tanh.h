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
    // 计算 tanh 所需的中间结果临时缓冲（TBuf 常驻 UB，不需要反复分配释放）
    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> tmpBuf3;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // 本核处理的元素数（尾核取剩余部分）
    int64_t ubLength_ = 0;     // UB 单次循环处理的元素数
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Init：绑定 GM 张量并初始化 UB 队列（Double Buffer）
//   blockLength_：totalNum 减去前面核已分配长度后与 blockFactor 取小，
//                 保证尾核不越界
//   ubLength_：   tiling 下发的单次 UB 循环元素数
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf0, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf3, ubLength_ * sizeof(T));
}

// CopyIn：把当前 tile 的 x 从 GM 搬入 UB 并入队
//   currentNum 为尾块实际元素数，DataCopyPad 可搬任意字节长度，避免尾块越界
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

// Compute：出队 x，执行 tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
//   矢量指令要求计算长度按 32 字节对齐（float 为 8 个元素、half 为 16 个），
//   不足时向上对齐计算，多余部分仍在 buffer 范围内，CopyOut 只回拷有效部分
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::LocalTensor<T> tmp0 = tmpBuf0.Get<T>();
    AscendC::LocalTensor<T> tmp1 = tmpBuf1.Get<T>();
    AscendC::LocalTensor<T> tmp2 = tmpBuf2.Get<T>();
    AscendC::LocalTensor<T> tmp3 = tmpBuf3.Get<T>();
    int64_t alignElems = 32 / sizeof(T);
    int64_t calcCount = ((currentNum + alignElems - 1) / alignElems) * alignElems;
    Exp(tmp0, xLocal, calcCount);           // exp(x)
    Muls(tmp1, xLocal, (T)-1.0, calcCount); // -x
    Exp(tmp2, tmp1, calcCount);             // exp(-x)
    Add(tmp3, tmp0, tmp2, calcCount);       // exp(x) + exp(-x)
    Sub(tmp1, tmp0, tmp2, calcCount);       // exp(x) - exp(-x)
    Div(yLocal, tmp1, tmp3, calcCount);     // tanh(x)
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// CopyOut：出队结果，从 UB 回拷 GM（只搬有效部分 currentNum）
template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

// Process：按 ubLength_ 切块循环，CopyIn → Compute → CopyOut
//   最后一块用剩余元素数 currentNum，避免尾块越界
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
