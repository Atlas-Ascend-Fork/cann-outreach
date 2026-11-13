/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 *
 * 平台「下载空工程」骨架版：TanhTilingData(totalNum / blockFactor / ubFactor)
 * + schMode 模板分支。Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;   // 输入/输出队列双缓冲

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

    // 题面要求「补充计算过程中所需的变量」：4 个 UB 临时缓冲
    TBuf<QuePosition::VECCALC> tmpBuf0;   // exp(x)
    TBuf<QuePosition::VECCALC> tmpBuf1;   // -x / 分子（复用）
    TBuf<QuePosition::VECCALC> tmpBuf2;   // exp(-x)
    TBuf<QuePosition::VECCALC> tmpBuf3;   // 分母 exp(x)+exp(-x)

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;   // 本核处理的元素数（尾核取剩余部分）
    int64_t ubLength_ = 0;      // UB 单次循环处理的元素数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ <= 0) {
        return;                                  // 无数据的核直接空转
    }
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

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    // DataCopyPad 支持任意字节长度，尾块不越界
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> tmp0 = tmpBuf0.Get<T>();
    LocalTensor<T> tmp1 = tmpBuf1.Get<T>();
    LocalTensor<T> tmp2 = tmpBuf2.Get<T>();
    LocalTensor<T> tmp3 = tmpBuf3.Get<T>();

    // 矢量指令计算长度需按 32 字节对齐；多算部分不回拷 GM
    int64_t alignElems = 32 / sizeof(T);
    int64_t calcCount = ((currentNum + alignElems - 1) / alignElems) * alignElems;

    // Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    AscendC::Exp(tmp0, xLocal, calcCount);                 // exp(x)
    AscendC::Muls(tmp1, xLocal, (T)-1.0, calcCount);      // -x
    AscendC::Exp(tmp2, tmp1, calcCount);                   // exp(-x)
    AscendC::Add(tmp3, tmp0, tmp2, calcCount);             // 分母 exp(x)+exp(-x)
    AscendC::Sub(tmp1, tmp0, tmp2, calcCount);             // 分子 exp(x)-exp(-x)（复用 tmp1）
    AscendC::Div(yLocal, tmp1, tmp3, calcCount);           // tanh(x)

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
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
