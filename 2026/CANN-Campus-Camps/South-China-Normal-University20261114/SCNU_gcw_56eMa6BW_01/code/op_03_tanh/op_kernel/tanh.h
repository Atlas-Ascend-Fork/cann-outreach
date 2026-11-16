
#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TMP_BUFFER_SIZE = 8192;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress, int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress, int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// 初始化
template <typename T>
__aicore__ inline void Tanh<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const TanhTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    int64_t offset =
        GetBlockIdx() * blockLength_;

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + offset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + offset,
        blockLength_);

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    // Tanh 高阶 API 使用的临时 UB 空间
    pipe.InitBuffer(
        tmpBuffer,
        TMP_BUFFER_SIZE);
}

// 将 GM 数据搬运到 UB
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        xLocal,
        inputGMX[progress * ubLength_],
        currentNum);

    inputQueueX.EnQue(xLocal);
}

// 执行 Tanh 计算
template <typename T>
__aicore__ inline void Tanh<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    LocalTensor<uint8_t> tmpLocal =
        tmpBuffer.Get<uint8_t>();

    AscendC::Tanh(
        yLocal,
        xLocal,
        tmpLocal,
        static_cast<uint32_t>(currentNum));

    outputQueueY.EnQue(yLocal);

    inputQueueX.FreeTensor(xLocal);
}

// 将计算结果搬回 GM
template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress * ubLength_],
        yLocal,
        currentNum);

    outputQueueY.FreeTensor(yLocal);
}

// 主处理流程
template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t loopCount =
        (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = ubLength_;

        if ((i + 1) * ubLength_ > blockLength_) {
            currentNum =
                blockLength_ - i * ubLength_;
        }

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh

#endif // TANH_H

