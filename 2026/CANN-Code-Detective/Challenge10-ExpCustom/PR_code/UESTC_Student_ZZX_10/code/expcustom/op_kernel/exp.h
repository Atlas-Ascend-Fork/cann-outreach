/*!
 * \file exp.h
 * \brief Exp operator kernel implementation.
 */

#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        const int64_t startOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
        const int64_t remainLength = tilingData->totalNum - startOffset;
        blockLength = remainLength < tilingData->blockFactor ? remainLength : tilingData->blockFactor;
        ubLength = tilingData->ubFactor;

        inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + startOffset, blockLength);
        outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + startOffset, blockLength);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (int64_t progress = 0; progress < blockLength; progress += ubLength) {
            const int64_t remainLength = blockLength - progress;
            const int64_t currentLength = remainLength < ubLength ? remainLength : ubLength;
            CopyIn(progress, currentLength);
            Compute(currentLength);
            CopyOut(progress, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentLength)
    {
        LocalTensor<DT_X> inputLocal = inputQueue.AllocTensor<DT_X>();
        DataCopyParams params{1, static_cast<uint16_t>(currentLength * sizeof(DT_X)), 0, 0};
        DataCopyPad(inputLocal, inputGM[progress], params, {false, 0, 0, 0});
        inputQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int64_t currentLength)
    {
        LocalTensor<DT_X> inputLocal = inputQueue.DeQue<DT_X>();
        LocalTensor<DT_X> outputLocal = outputQueue.AllocTensor<DT_X>();
        AscendC::Exp(outputLocal, inputLocal, currentLength);
        outputQueue.EnQue(outputLocal);
        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentLength)
    {
        LocalTensor<DT_X> outputLocal = outputQueue.DeQue<DT_X>();
        DataCopyParams params{1, static_cast<uint16_t>(currentLength * sizeof(DT_X)), 0, 0};
        DataCopyPad(outputGM[progress], outputLocal, params);
        outputQueue.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    GlobalTensor<DT_X> inputGM;
    GlobalTensor<DT_X> outputGM;
    int64_t blockLength = 0;
    int64_t ubLength = 0;
};

} // namespace NsExp
#endif // EXP_H
