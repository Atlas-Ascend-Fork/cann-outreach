/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义（in-place 双缓冲乒乓，读写重叠）
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t PIPE_NUM = 2;  // 乒乓缓冲块数

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    TPipe pipe;
    // 输入输出共用同一块 UB（in-place），两块乒乓：读 B 与 写 A 重叠
    TBuf<QuePosition::VECCALC> buf0;
    TBuf<QuePosition::VECCALC> buf1;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t blockFactor = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t gmOffset = blockFactor * blockIdx;
    int64_t remain = tilingData->totalNum - gmOffset;
    if (remain < 0) {
        remain = 0;
    }
    this->blockLength_ = (blockFactor < remain) ? blockFactor : remain;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, this->blockLength_);

    pipe.InitBuffer(buf0, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(buf1, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t progress = 0;
    int32_t iter = 0;
    while (progress < this->blockLength_) {
        int64_t currentNum = this->blockLength_ - progress;
        if (currentNum > this->ubLength_) {
            currentNum = this->ubLength_;
        }
        const uint32_t count = static_cast<uint32_t>(currentNum);
        const int32_t slot = iter & 1;
        const int32_t evt = (slot == 0) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<T> local = (slot == 0) ? buf0.Get<T>() : buf1.Get<T>();

        // 复用该缓冲前，等它上一次的搬出完成（流水第 3 次起）
        if (iter >= PIPE_NUM) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evt);
        }

        // GM -> UB
        AscendC::DataCopy(local, inputGMX[progress], count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);

        // in-place 计算
        AscendC::Relu(local, local, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evt);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evt);

        // UB -> GM（与下一轮的 GM->UB 在不同缓冲上并行）
        AscendC::DataCopy(outputGMY[progress], local, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evt);

        progress += currentNum;
        iter++;
    }

    // 收尾：等最后未消费的搬出完成标志
    if (iter >= 2) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    } else if (iter == 1) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

} // namespace NsRelu
#endif // RELU_H