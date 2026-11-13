/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 *
 * 计算公式：y = max(x, 0) + negativeSlope * min(x, 0)
 * 与 Sqrt/Abs 等单指令算子不同，需要临时 Tensor 完成 Max/Min/Mul/Add 复合计算
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// 单次 UB 计算分块上限：队列与临时 Tensor 最多占用 (2*2+2) * 4096 * 4B = 96KB，
// 为高阶 API 内部临时空间预留足够 UB 余量
constexpr int64_t MAX_UB_CHUNK = 4096;
// MTE 搬运的地址与长度对齐要求（字节）
constexpr int64_t COPY_ALIGN_BYTES = 32;

template <typename T>
class Leakyrelu {
public:
    __aicore__ inline Leakyrelu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData);
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
    TBuf<TPosition::VECCALC> zeroBuf; // 临时：0 向量，计算中复用为 α 向量
    TBuf<TPosition::VECCALC> negBuf;  // 临时：min(x, 0) * α

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    float negativeSlope_ = 0.0f;  // 属性：负半轴斜率 α（经 tilingData 下发）
    int64_t coreStartOffset_ = 0;  // 本核在全局数据中的起始偏移
    int64_t blockLength_ = 0;      // 本核需要处理的元素数量
    int64_t ubLength_ = 0;         // 实际每次 UB 循环处理的元素数量，不超过 MAX_UB_CHUNK
};

// 判断本段 GM 访问（起始地址与长度）是否满足 DataCopy 的 32B 对齐要求
template <typename T>
__aicore__ inline bool Leakyrelu<T>::IsGmAligned(int64_t gmOffset, int64_t currentNum) const
{
    return gmOffset * static_cast<int64_t>(sizeof(T)) % COPY_ALIGN_BYTES == 0 &&
           currentNum * static_cast<int64_t>(sizeof(T)) % COPY_ALIGN_BYTES == 0;
}

// Leakyrelu 算子核函数实现：CopyIn -> Compute -> CopyOut 三段式流水
template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    // 属性 negativeSlope 经 tilingData 下发到 kernel 侧
    negativeSlope_ = tilingData->negativeSlope;
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
        // 为复合计算分配临时 Tensor
        pipe.InitBuffer(zeroBuf, ubLength_ * sizeof(T));
        pipe.InitBuffer(negBuf, ubLength_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
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
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> zeroLocal = zeroBuf.Get<T>();
    LocalTensor<T> negLocal = negBuf.Get<T>();
    // 高阶 API 复合计算：y = max(x, 0) + α * min(x, 0)
    AscendC::Duplicate(zeroLocal, static_cast<T>(0), static_cast<uint32_t>(currentNum));   // zero = 0
    AscendC::Max(yLocal, xLocal, zeroLocal, static_cast<uint32_t>(currentNum));            // y = max(x, 0)
    AscendC::Min(negLocal, xLocal, zeroLocal, static_cast<uint32_t>(currentNum));         // neg = min(x, 0)
    AscendC::Duplicate(zeroLocal, static_cast<T>(negativeSlope_), static_cast<uint32_t>(currentNum)); // zero 复用为 α
    AscendC::Mul(negLocal, negLocal, zeroLocal, static_cast<uint32_t>(currentNum));       // neg = α * min(x, 0)
    AscendC::Add(yLocal, yLocal, negLocal, static_cast<uint32_t>(currentNum));            // y = max(x, 0) + α * min(x, 0)
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
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
__aicore__ inline void Leakyrelu<T>::Process()
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

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
