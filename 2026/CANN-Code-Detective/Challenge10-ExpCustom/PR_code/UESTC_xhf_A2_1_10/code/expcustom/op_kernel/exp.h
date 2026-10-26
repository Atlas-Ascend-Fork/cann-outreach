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

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        tileLength = tilingData->ubFactor;
        const int64_t blockOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
        const int64_t remaining = tilingData->totalNum - blockOffset;
        blockLength = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + blockOffset, blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + blockOffset, blockLength);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        const int64_t loopCount = (blockLength + tileLength - 1) / tileLength;
        for (int64_t i = 0; i < loopCount; ++i) {
            const int64_t remaining = blockLength - i * tileLength;
            const int64_t currentNum = remaining < tileLength ? remaining : tileLength;
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        // Host tiling guarantees blockFactor and ubFactor are 32-byte aligned.
        DataCopy(xLocal, inputGMX[progress * tileLength], currentNum);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Exp(yLocal, xLocal, currentNum);
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * tileLength], yLocal, currentNum);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    int64_t blockLength = 0;
    int64_t tileLength = 0;
};

} // namespace NsExp
#endif // EXP_H
