/*!
 * \file abs.h
 * \brief Abs 算子 kernel 类定义
 */
#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DMA_ALIGNMENT = 32;

template <typename T>
class Abs {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData)
    {
        const int64_t start = GetBlockIdx() * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - start;
        if (remaining <= 0) {
            return;
        }
        blockLength = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        tileLength = tilingData->ubFactor;
        inputGM.SetGlobalBuffer((__gm__ T*)x + start, blockLength);
        outputGM.SetGlobalBuffer((__gm__ T*)y + start, blockLength);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (int64_t offset = 0; offset < blockLength; offset += tileLength) {
            const int64_t remaining = blockLength - offset;
            const int32_t count = remaining < tileLength ? remaining : tileLength;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int32_t count)
    {
        LocalTensor<T> x = inputQueue.AllocTensor<T>();
        const uint32_t bytes = count * sizeof(T);
        if (bytes % DMA_ALIGNMENT == 0) {
            DataCopy(x, inputGM[offset], count);
        } else {
            const DataCopyExtParams copy{1, bytes, 0, 0, 0};
            const DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(x, inputGM[offset], copy, pad);
        }
        inputQueue.EnQue(x);
    }

    __aicore__ inline void Compute(int32_t count)
    {
        LocalTensor<T> x = inputQueue.DeQue<T>();
        LocalTensor<T> y = outputQueue.AllocTensor<T>();
        AscendC::Abs(y, x, count);
        outputQueue.EnQue(y);
        inputQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(int64_t offset, int32_t count)
    {
        LocalTensor<T> y = outputQueue.DeQue<T>();
        const uint32_t bytes = count * sizeof(T);
        if (bytes % DMA_ALIGNMENT == 0) {
            DataCopy(outputGM[offset], y, count);
        } else {
            const DataCopyExtParams copy{1, bytes, 0, 0, 0};
            DataCopyPad(outputGM[offset], y, copy);
        }
        outputQueue.FreeTensor(y);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;
    int64_t blockLength = 0;
    int64_t tileLength = 0;
};
} // namespace NsAbs
#endif
