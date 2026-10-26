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

constexpr int32_t BUFFER_NUM = 1; // One buffer per queue.

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        blockLength = tilingData->blockFactor;
        tileLength = tilingData->ubFactor;
        tileNum = blockLength / tileLength;
        const uint64_t offset = GetBlockIdx() * blockLength;
        inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + offset, blockLength);
        outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + offset, blockLength);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // Each iteration transfers and computes one complete tile.
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inputQueue.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGM[progress * tileLength], tileLength);
        inputQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueue.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueue.AllocTensor<DT_X>();
        AscendC::Exp(yLocal, xLocal, tileLength);
        outputQueue.EnQue(yLocal);
        inputQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outputQueue.DeQue<DT_X>();
        DataCopy(outputGM[progress * tileLength], yLocal, tileLength);
        outputQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    GlobalTensor<DT_X> inputGM;
    GlobalTensor<DT_X> outputGM;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
};

} // namespace NsExp
#endif // EXP_H
