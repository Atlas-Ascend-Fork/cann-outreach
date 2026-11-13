/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        const int64_t start = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
        if (start >= tilingData->totalNum) {
            return;
        }

        const int64_t remaining = tilingData->totalNum - start;
        blockLength = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        tileLength = static_cast<uint32_t>(tilingData->ubFactor);
        inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + start, blockLength);
        outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + start, blockLength);
    }

    __aicore__ inline void Process()
    {
        if (blockLength == 0) {
            return;
        }

        TPipe pipe;
        // 队列里同时只存一个待消费张量。每个队列可分配两个轮换使用的缓冲。
        const uint8_t bufferCount = blockLength > tileLength ? 2 : 1;
        const uint32_t tileBytes = tileLength * sizeof(T);
        pipe.InitBuffer(inputQueue, bufferCount, tileBytes);
        pipe.InitBuffer(outputQueue, bufferCount, tileBytes);

        for (int64_t offset = 0; offset < blockLength;) {
            const int64_t remaining = blockLength - offset;
            const uint32_t count = remaining < tileLength ? static_cast<uint32_t>(remaining) : tileLength;
            const uint32_t bytes = count * sizeof(T);
            const bool aligned = bytes % 32 == 0;
            CopyIn(offset, count, bytes, aligned);
            Compute(count);
            CopyOut(offset, count, bytes, aligned);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count, uint32_t bytes, bool aligned)
    {
        auto input = inputQueue.AllocTensor<T>();
        if (aligned) {
            DataCopy(input, inputGM[offset], count);
        } else {
            const DataCopyExtParams copy{1, bytes, 0, 0, 0};
            const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(input, inputGM[offset], copy, pad);
        }
        inputQueue.EnQue(input);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto input = inputQueue.DeQue<T>();
        auto output = outputQueue.AllocTensor<T>();
        AscendC::Relu(output, input, count);
        outputQueue.EnQue(output);
        inputQueue.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count, uint32_t bytes, bool aligned)
    {
        auto output = outputQueue.DeQue<T>();
        if (aligned) {
            DataCopy(outputGM[offset], output, count);
        } else {
            const DataCopyExtParams copy{1, bytes, 0, 0, 0};
            DataCopyPad(outputGM[offset], output, copy);
        }
        outputQueue.FreeTensor(output);
    }

    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;
    int64_t blockLength = 0;
    uint32_t tileLength = 0;
};

} // namespace NsRelu

#endif // RELU_H
