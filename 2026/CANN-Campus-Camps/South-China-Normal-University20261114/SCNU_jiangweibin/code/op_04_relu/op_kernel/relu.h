#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor;
        inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
        outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
            const int64_t remaining = blockLength_ - offset;
            const uint32_t count = remaining < ubLength_ ? remaining : ubLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count)
    {
        auto input = inputQueue_.AllocTensor<T>();
        DataCopy(input, inputGM_[offset], count);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto input = inputQueue_.DeQue<T>();
        auto output = outputQueue_.AllocTensor<T>();
        AscendC::Relu(output, input, count);
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count)
    {
        auto output = outputQueue_.DeQue<T>();
        DataCopy(outputGM_[offset], output, count);
        outputQueue_.FreeTensor(output);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    GlobalTensor<T> inputGM_, outputGM_;
    int64_t blockLength_, ubLength_;
};
} // namespace NsRelu
#endif
