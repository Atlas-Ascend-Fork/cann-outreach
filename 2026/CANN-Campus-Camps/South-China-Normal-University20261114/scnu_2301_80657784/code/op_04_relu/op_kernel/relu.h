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

        const int32_t bufferCount = blockLength_ <= ubLength_ ? 1 : BUFFER_NUM;
        pipe.InitBuffer(
            dataQueue,
            bufferCount,
            ubLength_ * sizeof(T)
        );
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }

        for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
            const int64_t count = GetTileCount(offset);
            // 同一个队列按块流转，双缓冲由 AllocTensor/FreeTensor 管理。
            CopyIn(offset, count);
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
        LocalTensor<T> xLocal = dataQueue.AllocTensor<T>();
        DataCopy(xLocal, inputGMX[offset], count);
        dataQueue.EnQue<QuePosition::GM, QuePosition::VECIN, T>(xLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        LocalTensor<T> xLocal =
            dataQueue.DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        // 源、目的地址完全相同，Relu 在原缓冲区中计算。
        AscendC::Relu(xLocal, xLocal, count);
        dataQueue.EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        LocalTensor<T> yLocal =
            dataQueue.DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
        DataCopy(outputGMY[offset], yLocal, count);
        dataQueue.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> dataQueue;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};
}  // namespace NsRelu
#endif  // RELU_H
