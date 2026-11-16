#ifndef RELU_H
#define RELU_H
#include "kernel_operator.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {
using namespace AscendC;

template<class T> class Relu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* d) {
        this->blockFactor = d->blockFactor;
        this->ubFactor = d->ubFactor;
        
        // 计算当前核的处理起点与长度
        uint64_t start = uint64_t(GetBlockIdx()) * d->blockFactor;
        if (start >= d->totalNum) return;
        uint64_t remaining = d->totalNum - start;
        this->coreLength = remaining < d->blockFactor ? remaining : d->blockFactor;

        xG.SetGlobalBuffer((__gm__ T*)x + start, this->coreLength);
        yG.SetGlobalBuffer((__gm__ T*)y + start, this->coreLength);

        // 初始化流水线，队列深度设为 2，开启自动双缓冲
        pipe.InitBuffer(inQueue, 2, this->ubFactor * sizeof(T));
        pipe.InitBuffer(outQueue, 2, this->ubFactor * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) return;
        
        uint32_t loopCount = (coreLength + ubFactor - 1) / ubFactor;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t currentFactor = (i == loopCount - 1) ? (coreLength - i * ubFactor) : ubFactor;
            CopyIn(i, currentFactor);
            Compute(currentFactor);
            CopyOut(i, currentFactor);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t index, uint32_t currentFactor) {
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        DataCopy(xLocal, xG[index * ubFactor], currentFactor);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t currentFactor) {
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();

        // ReLU: y = max(0, x)
        if constexpr (IsSameType<T, float>::value) {
            Maxs(yLocal, xLocal, 0.0f, currentFactor);
        } else {
            Maxs(yLocal, xLocal, (half)0.0, currentFactor);
        }

        outQueue.EnQue<T>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t index, uint32_t currentFactor) {
        LocalTensor<T> yLocal = outQueue.DeQue<T>();
        DataCopy(yG[index * ubFactor], yLocal, currentFactor);
        outQueue.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;   // 2 代表 DoubleBuffer
    TQue<QuePosition::VECOUT, 2> outQueue;
    GlobalTensor<T> xG, yG;
    uint32_t blockFactor = 0;
    uint32_t ubFactor = 0;
    uint32_t coreLength = 0;
};
} // namespace NsRelu
#endif