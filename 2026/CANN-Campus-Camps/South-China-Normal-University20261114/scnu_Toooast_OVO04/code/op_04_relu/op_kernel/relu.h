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
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData
    )
    {
        const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength_ = remaining < tilingData->blockFactor
            ? remaining
            : tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor;
        inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

        pipe.InitBuffer(
            inputQueueX,
            BUFFER_NUM,
            ubLength_ * sizeof(T)
        );
        pipe.InitBuffer(
            outputQueueY,
            BUFFER_NUM,
            ubLength_ * sizeof(T)
        );
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }

        CopyIn(0, GetTileCount(0));
        for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
            const int64_t nextOffset = offset + ubLength_;
            if (nextOffset < blockLength_) {
                CopyIn(nextOffset, GetTileCount(nextOffset));
            }
            const int64_t count = GetTileCount(offset);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline int64_t GetTileCount(int64_t offset)
    {
        const int64_t remaining = blockLength_ - offset;
        return remaining < ubLength_ ? remaining : ubLength_;
    }

    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        DataCopy(xLocal, inputGMX[offset], count);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
        AscendC::Relu(yLocal, xLocal, count);
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
        DataCopy(outputGMY[offset], yLocal, count);
        outputQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};
}  // namespace NsRelu
#endif  // RELU_H
