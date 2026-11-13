#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"
#include "lib/math/tanh.h"

namespace NsTanh {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t SCRATCH_BYTES = 32768;

template <typename T>
class Tanh {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor;
        inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
        outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe_.InitBuffer(scratch_, SCRATCH_BYTES);
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
        AscendC::Tanh(output, input, scratch_.Get<uint8_t>(), count);
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
    TBuf<QuePosition::VECCALC> scratch_;
    GlobalTensor<T> inputGM_, outputGM_;
    int64_t blockLength_, ubLength_;
};
} // namespace NsTanh
#endif
