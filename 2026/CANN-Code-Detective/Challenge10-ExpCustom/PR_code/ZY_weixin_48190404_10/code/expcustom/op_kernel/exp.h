#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

// 官方模板这里给的是 1，如果你的机器性能好，可以改成 2 开启双缓冲
constexpr int32_t BUFFER_NUM = 1; 

template <typename DT_X>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    GlobalTensor<DT_X> inputGM_;
    GlobalTensor<DT_X> outputGM_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename DT_X>
__aicore__ inline void Exp<DT_X>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;

    int64_t startOffset = blockIdx * blockFactor;
    if (startOffset >= totalNum) {
        blockLength_ = 0;
        return;
    }

    blockLength_ = (startOffset + blockFactor <= totalNum) ? blockFactor : (totalNum - startOffset);
    ubLength_ = tilingData->ubFactor;

    inputGM_.SetGlobalBuffer((__gm__ DT_X*)x + startOffset, blockLength_);
    outputGM_.SetGlobalBuffer((__gm__ DT_X*)y + startOffset, blockLength_);

    if (blockLength_ > 0) {
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, ubLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, ubLength_ * sizeof(DT_X));
    }
}

template <typename DT_X>
__aicore__ inline void Exp<DT_X>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<DT_X> xLocal = inputQueue_.AllocTensor<DT_X>();
    DataCopy(xLocal, inputGM_[progress], currentNum);
    inputQueue_.EnQue(xLocal);
}

template <typename DT_X>
__aicore__ inline void Exp<DT_X>::Compute(int64_t currentNum)
{
    LocalTensor<DT_X> xLocal = inputQueue_.DeQue<DT_X>();
    LocalTensor<DT_X> yLocal = outputQueue_.AllocTensor<DT_X>();

    // 注意：使用 AscendC::Exp 避免和当前类名 Exp 冲突
    AscendC::Exp(yLocal, xLocal, currentNum);

    outputQueue_.EnQue(yLocal);
    inputQueue_.FreeTensor(xLocal);
}

template <typename DT_X>
__aicore__ inline void Exp<DT_X>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<DT_X> yLocal = outputQueue_.DeQue<DT_X>();
    DataCopy(outputGM_[progress], yLocal, currentNum);
    outputQueue_.FreeTensor(yLocal);
}

template <typename DT_X>
__aicore__ inline void Exp<DT_X>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t progress = 0;
    while (progress < blockLength_) {
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);

        progress += currentNum;
    }
}

} // namespace NsExp
#endif // EXP_H