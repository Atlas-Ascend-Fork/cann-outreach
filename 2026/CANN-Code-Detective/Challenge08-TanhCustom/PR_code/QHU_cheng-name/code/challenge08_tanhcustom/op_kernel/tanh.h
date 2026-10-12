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
constexpr int64_t kCopyBlockBytes = 32;
constexpr int64_t kUbBankSkew = 256;
constexpr int64_t kUbLongLoopSkew = 512;
constexpr int64_t kLongLoopSkewStart = 4;

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
__aicore__ inline int64_t LessOf(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
__aicore__ inline bool CanUsePlainCopy(int64_t todo, int64_t typeSize)
{
    return (todo * typeSize) % kCopyBlockBytes == 0;
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
    __aicore__ inline void RunCompactTile();
    __aicore__ inline void ComputeTanh(AscendC::LocalTensor<T> dstLocal,
                                        AscendC::LocalTensor<T> xLocal,
                                        AscendC::LocalTensor<T> scratchLocal,
                                        int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> inputBufX;
    TBuf<TPosition::VECCALC> outputBufY;
    TBuf<TPosition::VECCALC> workBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t activeElems_ = 0; 
    int64_t chunkElems_ = 0;
    bool compactTile_ = false;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t coreId = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t coreOffset = tilingData->blockFactor * coreId;
    int64_t restElems = tilingData->totalNum - coreOffset;
    activeElems_ = restElems > tilingData->blockFactor ? tilingData->blockFactor : restElems;
    if (activeElems_ < 0) {
        activeElems_ = 0;
    }
    chunkElems_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T *)x + coreOffset, activeElems_);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + coreOffset, activeElems_);
    int64_t loopCount = (activeElems_ + chunkElems_ - 1) / chunkElems_;
    compactTile_ = (loopCount <= 1);
    uint8_t queueBlocks = (loopCount >= 2) ? 2 : 1;
    int64_t bankSkew = 0;
    if (loopCount >= kLongLoopSkewStart) {
        bankSkew = kUbLongLoopSkew;
    } else if (loopCount >= 2) {
        bankSkew = kUbBankSkew;
    }
    int64_t ubChunkBytes = chunkElems_ * sizeof(T) + bankSkew;
    if (compactTile_) {
        pipe.InitBuffer(inputBufX, ubChunkBytes);
        pipe.InitBuffer(outputBufY, ubChunkBytes);
    } else {
        pipe.InitBuffer(inputQueueX, queueBlocks, ubChunkBytes);
        pipe.InitBuffer(outputQueueY, queueBlocks, ubChunkBytes);
    }
    pipe.InitBuffer(workBuf, ubChunkBytes);
}

template <typename T>
__aicore__ inline void Tanh<T>::ComputeTanh(AscendC::LocalTensor<T> dstLocal,
                                        AscendC::LocalTensor<T> srcLocal,
                                        AscendC::LocalTensor<T> workLocal,
                                        int64_t currentNum){
    // 优化实现：tanh(x) = 1 - 2/(e^{2x}+1)
    // 仅 1 次 Exp；数值稳定（x→±∞ 均收敛 ±1，无 NaN）
    AscendC::Muls(workLocal, srcLocal, (T)2.0f, currentNum);     // 2x
    AscendC::Exp(workLocal, workLocal, currentNum);               // e^{2x}
    AscendC::Adds(workLocal, workLocal, (T)1.0f, currentNum);     // e^{2x}+1
    AscendC::Duplicate(dstLocal, (T)2.0f, currentNum);            // 常量 2
    AscendC::Div(dstLocal, dstLocal, workLocal, currentNum);      // 2/(e^{2x}+1)
    AscendC::Muls(dstLocal, dstLocal, (T)-1.0f, currentNum);      // -2/(e^{2x}+1)
    AscendC::Adds(dstLocal, dstLocal, (T)1.0f, currentNum);       // 1 - 2/(e^{2x}+1)
}
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    AscendC::LocalTensor<T> srcLocal = inputQueueX.template AllocTensor<T>();
    // 对齐时走 DataCopy 快速路径，非对齐才退 DataCopyPad
    if (CanUsePlainCopy(currentNum, sizeof(T))) {
        AscendC::DataCopy(srcLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    } else {
        AscendC::DataCopyParams copySpec;
        copySpec.blockCount = 1;
        copySpec.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copySpec.srcStride = 0;
        copySpec.dstStride = 0;
        AscendC::DataCopyPad(srcLocal, inputGMX[progress], copySpec, {false, 0, 0, 0});
    }
    inputQueueX.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    AscendC::LocalTensor<T> srcLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> dstLocal = outputQueueY.template AllocTensor<T>();
    AscendC::LocalTensor<T> workLocal = workBuf.template Get<T>();

    ComputeTanh(dstLocal,srcLocal,workLocal,currentNum);

    outputQueueY.template EnQue<T>(dstLocal);
    inputQueueX.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    AscendC::LocalTensor<T> dstLocal = outputQueueY.template DeQue<T>();
    // 对齐时走 DataCopy 快速路径
    if (CanUsePlainCopy(currentNum, sizeof(T))) {
        AscendC::DataCopy(outputGMY[progress], dstLocal, static_cast<uint32_t>(currentNum));
    } else {
        AscendC::DataCopyParams copySpec;
        copySpec.blockCount = 1;
        copySpec.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copySpec.srcStride = 0;
        copySpec.dstStride = 0;
        AscendC::DataCopyPad(outputGMY[progress], dstLocal, copySpec);
    }
    outputQueueY.FreeTensor(dstLocal);
}
template <typename T>
__aicore__ inline void Tanh<T>::RunCompactTile(){
    int64_t todo = activeElems_;
    AscendC::LocalTensor<T> srcLocal = inputBufX.template Get<T>();
    AscendC::LocalTensor<T> dstLocal = outputBufY.template Get<T>();
    AscendC::LocalTensor<T> workLocal = workBuf.template Get<T>();

    AscendC::DataCopyParams copySpec;
    copySpec.blockCount = 1;
    copySpec.blockLen = static_cast<uint32_t>(todo * sizeof(T));
    copySpec.srcStride = 0;
    copySpec.dstStride = 0;

    AscendC::DataCopyPad(srcLocal, inputGMX[0], copySpec, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    ComputeTanh(dstLocal,srcLocal,workLocal,todo);

    AscendC::PipeBarrier<PIPE_ALL>();

    // 尾块安全：非 32B 对齐时必须走 DataCopyPad（按字节精确搬运，无越界），
    // 对齐时才允许走更快的 DataCopy——两种路径均不产生 GM 越界读写
    if (CanUsePlainCopy(todo, sizeof(T))) {
        AscendC::DataCopy(outputGMY[0], dstLocal, static_cast<uint32_t>(todo));
    } else {
        AscendC::DataCopyPad(outputGMY[0], dstLocal, copySpec);
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{ 
    // TODO: 实现 Process 逻辑
    if (activeElems_ <= 0) {
        return;
    }

    if (compactTile_) {
        RunCompactTile();
        return;
    }

    int64_t loopCount = (activeElems_ + chunkElems_ - 1) / chunkElems_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t tileOffset = i * chunkElems_;
        int64_t todo = LessOf(chunkElems_, activeElems_ - tileOffset);
        CopyIn(tileOffset, todo);
        Compute(todo);
        CopyOut(tileOffset, todo);
    }
}
} // namespace NsTanh
#endif // TANH_H
