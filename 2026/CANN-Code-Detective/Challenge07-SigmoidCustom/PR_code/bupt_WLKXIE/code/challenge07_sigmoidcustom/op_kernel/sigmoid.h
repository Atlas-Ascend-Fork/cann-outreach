/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {
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
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void RunCompactTile();
    __aicore__ inline void ComputeSigmoid(AscendC::LocalTensor<T> dstLocal,
                                        AscendC::LocalTensor<T> xLocal,
                                        int64_t currentNum);


private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> inputBufX;
    TBuf<TPosition::VECCALC> outputBufY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t activeElems_ = 0; 
    int64_t chunkElems_ = 0;
    bool compactTile_ = false;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    int64_t coreId = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t coreOffset = tilingData->core_elems * coreId;
    int64_t restElems = tilingData->total_elems - coreOffset;
    activeElems_ = restElems > tilingData->core_elems ? tilingData->core_elems : restElems;
    if (activeElems_ < 0) {
        activeElems_ = 0;
    }
    chunkElems_ = tilingData->tile_elems;
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
}

template <typename T>
__aicore__ inline void Sigmoid<T>::ComputeSigmoid(AscendC::LocalTensor<T> dstLocal,
                                        AscendC::LocalTensor<T> srcLocal,
                                        int64_t currentNum){
    T one = 1.0;
    AscendC::Exp(srcLocal,srcLocal,currentNum); // Exp(x) -> srcLocal
    AscendC::Adds(dstLocal,srcLocal,one,currentNum); //1+Exp(x) -> dstLocal
    AscendC::Div(dstLocal,srcLocal,dstLocal,currentNum);//Exp(x)/[1+Exp(x)] -> dstLocal
}
template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    AscendC::LocalTensor<T> srcLocal = inputQueueX.template AllocTensor<T>();
    AscendC::DataCopyParams copySpec;
    copySpec.blockCount = 1;
    copySpec.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copySpec.srcStride = 0;
    copySpec.dstStride = 0;
    AscendC::DataCopyPad(srcLocal, inputGMX[progress], copySpec, {false, 0, 0, 0});
    inputQueueX.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    AscendC::LocalTensor<T> srcLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> dstLocal = outputQueueY.template AllocTensor<T>();

    ComputeSigmoid(dstLocal,srcLocal,currentNum);

    outputQueueY.template EnQue<T>(dstLocal);
    inputQueueX.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    AscendC::LocalTensor<T> dstLocal = outputQueueY.template DeQue<T>();
    AscendC::DataCopyParams copySpec;
    copySpec.blockCount = 1;
    copySpec.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copySpec.srcStride = 0;
    copySpec.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress], dstLocal, copySpec);
    outputQueueY.FreeTensor(dstLocal);
}
template <typename T>
__aicore__ inline void Sigmoid<T>::RunCompactTile(){
    int64_t todo = activeElems_;
    AscendC::LocalTensor<T> srcLocal = inputBufX.template Get<T>();
    AscendC::LocalTensor<T> dstLocal = outputBufY.template Get<T>();

    AscendC::DataCopyParams copySpec;
    copySpec.blockCount = 1;
    copySpec.blockLen = static_cast<uint32_t>(todo * sizeof(T));
    copySpec.srcStride = 0;
    copySpec.dstStride = 0;

    AscendC::DataCopyPad(srcLocal, inputGMX[0], copySpec, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    ComputeSigmoid(dstLocal,srcLocal,todo);

    AscendC::PipeBarrier<PIPE_ALL>();

    if (CanUsePlainCopy(todo, sizeof(T))) {
        AscendC::DataCopy(outputGMY[0], dstLocal, static_cast<uint32_t>(todo));
    } else {
        AscendC::DataCopyPad(outputGMY[0], dstLocal, copySpec);
    }
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
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

} // namespace NsSigmoid
#endif // SIGMOID_H
