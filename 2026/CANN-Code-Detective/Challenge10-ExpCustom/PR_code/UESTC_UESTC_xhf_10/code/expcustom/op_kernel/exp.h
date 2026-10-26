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

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        if (tilingData->totalNum == 0) {
            this->blockLength = 0;
            return;
        }
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t begin = blockIdx * tilingData->blockFactor;
        int64_t remain = tilingData->totalNum - begin;
        if (remain < 0) {
            remain = 0;
        }
        this->blockLength = remain < tilingData->blockFactor ? remain : tilingData->blockFactor;
        this->tileLength = tilingData->ubFactor;

        inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + begin, this->blockLength);
        outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + begin, this->blockLength);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength <= 0) {
            return;
        }
        const int64_t loopCount = (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t currentLength = this->blockLength - i * this->tileLength;
            currentLength = currentLength < this->tileLength ? currentLength : this->tileLength;
            CopyIn(i, currentLength);
            Compute(currentLength);
            CopyOut(i, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentLength)
    {
        LocalTensor<DT_X> xLocal = inputQueue.AllocTensor<DT_X>();
        DataCopyExtParams params{1, static_cast<uint32_t>(currentLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGM[progress * this->tileLength], params, padParams);
        inputQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentLength)
    {
        LocalTensor<DT_X> xLocal = inputQueue.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueue.AllocTensor<DT_X>();
        AscendC::Exp(yLocal, xLocal, currentLength);
        outputQueue.EnQue(yLocal);
        inputQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentLength)
    {
        LocalTensor<DT_X> yLocal = outputQueue.DeQue<DT_X>();
        DataCopyExtParams params{1, static_cast<uint32_t>(currentLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(outputGM[progress * this->tileLength], yLocal, params);
        outputQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    GlobalTensor<DT_X> inputGM;
    GlobalTensor<DT_X> outputGM;
    int64_t blockLength = 0;
    int64_t tileLength = 0;
};

} // namespace NsExp
#endif // EXP_H
