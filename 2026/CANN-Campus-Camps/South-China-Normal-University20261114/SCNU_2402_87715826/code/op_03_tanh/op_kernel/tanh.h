/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "lib/math/tanh.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    static constexpr uint32_t TMP_BYTES = 32 * 1024;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuffer, TMP_BYTES);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto xLocal = inputQueueX.template AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    auto xLocal = inputQueueX.template DeQue<T>();
    auto yLocal = outputQueueY.template AllocTensor<T>();
    AscendC::Tanh(yLocal, xLocal, tmpBuffer.Get<uint8_t>(), currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto yLocal = outputQueueY.template DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    for (int64_t offset = 0, progress = 0; offset < blockLength_; offset += ubLength_, ++progress) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
