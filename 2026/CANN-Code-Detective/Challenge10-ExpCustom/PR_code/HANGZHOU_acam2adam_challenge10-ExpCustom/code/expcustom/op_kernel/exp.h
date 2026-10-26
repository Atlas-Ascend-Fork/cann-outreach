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
// 单块 UB tensor 的字节上限：双缓冲下 (输入 2 块 + 输出 2 块) 共 4 块，
// 控制在 32KB/块可保证总占用不超过 UB 容量
constexpr int64_t MAX_TILE_BYTES = 32 * 1024;
constexpr int64_t UB_BLOCK_BYTES = 32;

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
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    const int64_t blockIdx = AscendC::GetBlockIdx();
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t remainderLength = tilingData->totalNum - blockFactor * blockIdx;
    blockLength_ = (remainderLength > blockFactor) ? blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    // ubFactor 由 host 按 UB 容量给出；若过大（如小 shape / 手工构造的 tiling），
    // 在 kernel 内自动拆分为更小的 tile，保证双缓冲下 UB 不超限
    const int64_t typeSize = static_cast<int64_t>(sizeof(T));
    const int64_t alignElem = UB_BLOCK_BYTES / typeSize;
    const int64_t maxTileElem = MAX_TILE_BYTES / typeSize;
    int64_t tileElem = tilingData->ubFactor;
    if (tileElem <= 0 || tileElem > maxTileElem) {
        tileElem = maxTileElem;
    }
    ubLength_ = (tileElem / alignElem) * alignElem;
    if (ubLength_ <= 0) {
        ubLength_ = alignElem;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockFactor * blockIdx);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockFactor * blockIdx);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * typeSize);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * typeSize);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
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
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();
    AscendC::Exp(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
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
__aicore__ inline void Exp<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        const int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsExp
#endif // EXP_H
