/*!
 * \file sqrt.h
 * \brief Sqrt vector kernel with bounded tail copies and double buffering.
 */
#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace NsSqrt {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Sqrt {
public:
    __aicore__ inline Sqrt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
    {
        const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength_ = remaining > 0 ?
            (remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor) : 0;
        ubLength_ = tilingData->ubFactor;
        if (blockLength_ == 0 || ubLength_ <= 0) {
            return;
        }
        inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
        outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (ubLength_ <= 0) {
            return;
        }
        for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
            const int64_t remaining = blockLength_ - offset;
            const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
            CopyIn(offset, currentNum);
            Compute(currentNum);
            CopyOut(offset, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum)
    {
        LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        if (currentNum * sizeof(T) % 32 == 0) {
            DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
        } else {
            const DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
            const DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
        AscendC::Sqrt(yLocal, xLocal, static_cast<uint32_t>(currentNum));
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum)
    {
        LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
        if (currentNum * sizeof(T) % 32 == 0) {
            DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
        } else {
            const DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
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
} // namespace NsSqrt
#endif
