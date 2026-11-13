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

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remainingNum = tilingData->totalNum - blockOffset;
    blockLength_ = remainingNum < tilingData->blockFactor ? remainingNum : tilingData->blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ == 0 || ubLength_ == 0) {
        return;
    }

    // ReLU 不需要保留原始输入，输入输出原地复用同一块 UB；
    // Ping/Pong 两块缓冲即可支撑双缓冲流水，单块容量取 UB 的一半。
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    LocalTensor<T> ping(TPosition::VECCALC, 0, ubLength_);
    LocalTensor<T> pong(TPosition::VECCALC, bufferBytes, ubLength_);

    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t offset = progress * ubLength_;
        const int64_t remainingNum = blockLength_ - offset;
        const int64_t currentNum = remainingNum < ubLength_ ? remainingNum : ubLength_;
        const int32_t eventID = ((progress & 1) == 0) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<T>& local = ((progress & 1) == 0) ? ping : pong;

        // 等上一轮写回释放该缓冲，再搬入本块。
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        DataCopy(local, inputGMX[offset], currentNum);
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);

        Maxs(local, local, static_cast<T>(0), currentNum);

        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        DataCopy(outputGMY[offset], local, currentNum);
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

} // namespace NsRelu
#endif // RELU_H
