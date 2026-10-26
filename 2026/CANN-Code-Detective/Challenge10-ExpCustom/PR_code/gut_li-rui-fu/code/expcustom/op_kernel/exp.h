/*!
 * \file exp.h
 * \brief Exp 算子 kernel 类定义
 */

#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// 单次 UB 计算分块上限：队列最多占用 4 * 4096 * 4B = 64KB，
// 为高阶 API 内部临时空间预留足够 UB 余量
constexpr int64_t MAX_UB_CHUNK = 4096;
// MTE 搬运的地址与长度对齐要求（字节）
constexpr int64_t COPY_ALIGN_BYTES = 32;

template <typename T>
class Exp {
public:
    __aicore__ inline Exp(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    // 判断本段 GM 访问（起始地址与长度）是否满足 DataCopy 的 32B 对齐要求
    __aicore__ inline bool IsGmAligned(int64_t gmOffset, int64_t currentNum) const;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t coreStartOffset_ = 0;  // 本核在全局数据中的起始偏移
    int64_t blockLength_ = 0;      // 本核需要处理的元素数量
    int64_t ubLength_ = 0;         // 实际每次 UB 循环处理的元素数量，不超过 MAX_UB_CHUNK
};

// 判断本段 GM 访问（起始地址与长度）是否满足 DataCopy 的 32B 对齐要求
template <typename T>
__aicore__ inline bool Exp<T>::IsGmAligned(int64_t gmOffset, int64_t currentNum) const
{
    return gmOffset * static_cast<int64_t>(sizeof(T)) % COPY_ALIGN_BYTES == 0 &&
           currentNum * static_cast<int64_t>(sizeof(T)) % COPY_ALIGN_BYTES == 0;
}

// Exp 算子核函数实现：CopyIn -> Compute -> CopyOut 三段式流水
template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    inputGMX.SetGlobalBuffer((__gm__ T*)x);
    outputGMY.SetGlobalBuffer((__gm__ T*)y);

    // 多核数据切分：按 GetBlockIdx() 计算本核在全局数据中的起始偏移
    coreStartOffset_ = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;
    if (coreStartOffset_ >= tilingData->totalNum) {
        // 尾核无数据需要处理
        blockLength_ = 0;
        return;
    }
    int64_t remainNum = tilingData->totalNum - coreStartOffset_;
    blockLength_ = remainNum < tilingData->blockFactor ? remainNum : tilingData->blockFactor;

    if (tilingData->ubFactor > 0) {
        // 分块向上取整到 32B 对齐的元素数，保证队列内每个缓冲区起始地址 32B 对齐
        int64_t alignElems = COPY_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
        int64_t rawChunk = tilingData->ubFactor < MAX_UB_CHUNK ? tilingData->ubFactor : MAX_UB_CHUNK;
        ubLength_ = (rawChunk + alignElems - 1) / alignElems * alignElems;
        // 为队列分配 UB 内存（双缓冲）
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // 对齐时走 DataCopy 快路径；未对齐时回退 DataCopyPad，保证任意形状/偏移下的正确性
    int64_t gmOffset = coreStartOffset_ + progress * ubLength_;
    if (IsGmAligned(gmOffset, currentNum)) {
        DataCopy(xLocal, inputGMX[gmOffset], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[gmOffset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 高阶 API：逐元素计算 y = e^x
    AscendC::Exp(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t gmOffset = coreStartOffset_ + progress * ubLength_;
    if (IsGmAligned(gmOffset, currentNum)) {
        DataCopy(outputGMY[gmOffset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(outputGMY[gmOffset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    // 按 ubLength_ 分块处理本核数据，尾块长度可能小于 ubLength_；双缓冲流水：相邻循环的 CopyIn 与 CopyOut 交替并行
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t remainNum = blockLength_ - i * ubLength_;
        int64_t currentNum = remainNum < ubLength_ ? remainNum : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsExp
#endif // EXP_H
