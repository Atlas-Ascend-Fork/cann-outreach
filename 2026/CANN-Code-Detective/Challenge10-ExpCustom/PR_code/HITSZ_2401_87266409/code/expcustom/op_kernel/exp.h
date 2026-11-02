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

template <typename T>
class Exp {
public:
    __aicore__ inline Exp(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t offset = blockIdx * tilingData->blockFactor;
        int64_t remaining = tilingData->totalNum - offset;
        this->blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        if (this->blockLength_ < 0) {
            this->blockLength_ = 0;
        }
        this->ubLength_ = tilingData->ubFactor;
        if (this->ubLength_ < 1) {
            this->ubLength_ = 1;
        }

        // 每个核负责 blockFactor 个元素，最后一个核按剩余元素数收尾
        inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, this->blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, this->blockLength_);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t currentNum = this->ubLength_;
            if (i == loopCount - 1) {
                currentNum = this->blockLength_ - i * this->ubLength_;
            }
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
        // y = e^x
        AscendC::Exp(yLocal, xLocal, currentNum);
        outputQueueY.EnQue<T>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
        AscendC::DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0; // 当前核实际处理的元素数
    int64_t ubLength_ = 0;    // 单次 UB 循环处理的元素数
};

} // namespace NsExp
#endif // EXP_H
