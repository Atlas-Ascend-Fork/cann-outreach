/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

template <typename T, bool SMALL>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, uint32_t currentNum);
    __aicore__ inline void Compute(uint32_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, uint32_t currentNum);
    __aicore__ inline void ComputeAndCopyOut(int64_t progress, uint32_t currentNum);
    __aicore__ inline void ProcessSmall();

private:
    // 小/大 shape 都用双缓冲：小 shape 拆成多个 tile 让搬运与计算重叠
    static constexpr int32_t BUF_NUM = 2;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUF_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUF_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> calcBufSmall;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t actualLength_ = 0;
};

template <typename T, bool SMALL>
__aicore__ inline void Relu<T, SMALL>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    this->totalNum_ = tilingData->totalNum;
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    // 多核切分：每核 blockFactor 个元素，最后一核只处理剩余元素
    int64_t offset = this->blockLength_ * static_cast<int64_t>(GetBlockIdx());
    if (offset >= this->totalNum_) {
        this->actualLength_ = 0;
        return;
    }
    this->actualLength_ = this->totalNum_ - offset;
    if (this->actualLength_ > this->blockLength_) {
        this->actualLength_ = this->blockLength_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, this->actualLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, this->actualLength_);

    if constexpr (SMALL) {
        // 小 shape：单 TBuf + 手动事件，避免两个 TQue 的初始化与运行时分配
        pipe.InitBuffer(calcBufSmall, this->blockLength_ * sizeof(T));
    } else {
        pipe.InitBuffer(inputQueueX, BUF_NUM, this->ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUF_NUM, this->ubLength_ * sizeof(T));
    }
}

template <typename T, bool SMALL>
__aicore__ inline void Relu<T, SMALL>::CopyIn(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], currentNum);
    inputQueueX.template EnQue<T>(xLocal);
}

template <typename T, bool SMALL>
__aicore__ inline void Relu<T, SMALL>::Compute(uint32_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, currentNum);
    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, bool SMALL>
__aicore__ inline void Relu<T, SMALL>::CopyOut(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T, bool SMALL>
__aicore__ inline void Relu<T, SMALL>::ComputeAndCopyOut(int64_t progress, uint32_t currentNum)
{
    // 小 shape 专用：原地计算，省掉 output queue 的初始化与搬运
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::Relu(xLocal, xLocal, currentNum);
    event_t evt = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evt);
    WaitFlag<HardEvent::V_MTE3>(evt);
    DataCopy(outputGMY[progress], xLocal, currentNum);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, bool SMALL>
__aicore__ inline void Relu<T, SMALL>::ProcessSmall()
{
    uint32_t len = static_cast<uint32_t>(this->actualLength_);
    LocalTensor<T> xLocal = calcBufSmall.Get<T>();
    DataCopy(xLocal, inputGMX, len);

    event_t mte2v = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(mte2v);
    WaitFlag<HardEvent::MTE2_V>(mte2v);

    AscendC::Relu(xLocal, xLocal, len);

    event_t vmte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vmte3);
    WaitFlag<HardEvent::V_MTE3>(vmte3);

    DataCopy(outputGMY, xLocal, len);
}

template <typename T, bool SMALL>
__aicore__ inline void Relu<T, SMALL>::Process()
{
    if (this->actualLength_ <= 0) {
        return;
    }
    if constexpr (SMALL) {
        ProcessSmall();
        return;
    }
    int64_t progress = 0;
    int64_t remaining = this->actualLength_;
    while (remaining > 0) {
        int64_t current = remaining < this->ubLength_ ? remaining : this->ubLength_;
        CopyIn(progress, static_cast<uint32_t>(current));
        Compute(static_cast<uint32_t>(current));
        CopyOut(progress, static_cast<uint32_t>(current));
        progress += current;
        remaining -= current;
    }
}

} // namespace NsRelu
#endif // RELU_H
