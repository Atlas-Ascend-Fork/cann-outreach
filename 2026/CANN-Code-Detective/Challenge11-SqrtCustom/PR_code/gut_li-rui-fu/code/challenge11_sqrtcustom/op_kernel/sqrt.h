/*!
 * \file sqrt.h
 * \brief Sqrt 算子 kernel 类定义
 *
 * 逐元素平方根：y[i] = sqrt(x[i])，x[i] >= 0。
 * 采用 GM -> UB -> GM 的三段式流水（CopyIn / Compute / CopyOut），
 * 每个队列开双缓冲（BUFFER_NUM = 2），使搬运与计算重叠。
 */

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace NsSqrt {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Sqrt {
public:
    __aicore__ inline Sqrt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData);
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
__aicore__ inline void Sqrt<T>::Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    // 本核负责的数据区间：[blockIdx * blockFactor, blockIdx * blockFactor + blockLength_)
    // 尾核不足 blockFactor 时按剩余量截断；blockFactor 已由 host 侧按 32B 对齐，
    // 保证各核在 GM 上的起始地址均 32B 对齐，避免 DataCopyPad 写出越界覆盖邻核数据。
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    // 单次 UB 循环的处理长度；防御非法 tiling（0 长度会导致 InitBuffer 申请 0 字节）
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    // 按核偏移切分 GM，长度限定为本核 blockLength_
    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * blockIdx, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * blockIdx, blockLength_);

    // 输入/输出各 BUFFER_NUM 块，用于双缓冲流水
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();

    // 用 DataCopyPad 而非 DataCopy：尾块 currentNum 不保证 32B 对齐，
    // DataCopyPad 按字节搬运并自动补齐/裁剪，无需在 kernel 侧手工对齐。
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    // 注意：类名 Sqrt 会遮蔽 AscendC::Sqrt，必须显式带上 AscendC:: 命名空间，
    // 否则此处会被解析成构造函数调用。Sqrt 支持 half / float（dav_c220 起）。
    AscendC::Sqrt<T>(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Process()
{
    // 每轮处理 ubLength_ 个元素，尾轮按剩余量收窄
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        const int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSqrt
#endif // SQRT_H
