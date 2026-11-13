/*!
 * \file abs.h
 * \brief Abs 算子 kernel 类定义
 */

#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Abs {
public:
    __aicore__ inline Abs(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Abs<T>::Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData)
{
    const int64_t start = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - start;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + start, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + start, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Abs<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> input = inputQueueX.AllocTensor<T>();
    DataCopyParams params{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
    DataCopyPad(input, inputGMX[progress], params, {false, 0, 0, 0});
    inputQueueX.EnQue(input);
}

template <typename T>
__aicore__ inline void Abs<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> input = inputQueueX.DeQue<T>();
    LocalTensor<T> output = outputQueueY.AllocTensor<T>();
    AscendC::Abs(output, input, currentNum);
    outputQueueY.EnQue(output);
    inputQueueX.FreeTensor(input);
}

template <typename T>
__aicore__ inline void Abs<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> output = outputQueueY.DeQue<T>();
    DataCopyParams params{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
    DataCopyPad(outputGMY[progress], output, params);
    outputQueueY.FreeTensor(output);
}

template <typename T>
__aicore__ inline void Abs<T>::Process()
{
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }
}

} // namespace NsAbs
#endif // ABS_H

