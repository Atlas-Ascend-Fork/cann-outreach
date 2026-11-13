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

constexpr uint32_t COPY_ALIGN_BYTES = 32;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <bool Aligned>
    __aicore__ inline void ProcessTile(LocalTensor<T>& local, int64_t offset, uint32_t count, int32_t eventId);

    template <bool Aligned>
    __aicore__ inline void ProcessTiles();

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    AscendC::InitSocState();
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (blockOffset >= tilingData->totalNum || tilingData->ubFactor <= 0) {
        return;
    }

    const int64_t remaining = tilingData->totalNum - blockOffset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset, blockLength_);
}

template <typename T>
template <bool Aligned>
__aicore__ inline void Relu<T>::ProcessTile(LocalTensor<T>& local, int64_t offset, uint32_t count, int32_t eventId)
{
    if constexpr (Aligned) {
        DataCopy(local, inputGMX[offset], count);
    } else {
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, inputGMX[offset], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    AscendC::Relu(local, local, static_cast<int32_t>(count));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    if constexpr (Aligned) {
        DataCopy(outputGMY[offset], local, count);
    } else {
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], local, copyParams);
    }
}

template <typename T>
template <bool Aligned>
__aicore__ inline void Relu<T>::ProcessTiles()
{
    const uint32_t tileElements = static_cast<uint32_t>(ubLength_);
    LocalTensor<T> ping(TPosition::VECCALC, 0, tileElements);
    if (blockLength_ <= ubLength_) {
        ProcessTile<Aligned>(ping, 0, static_cast<uint32_t>(blockLength_), EVENT_ID0);
        return;
    }

    LocalTensor<T> pong(TPosition::VECCALC, static_cast<uint32_t>(tileElements * sizeof(T)), tileElements);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    bool usePing = true;
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const uint32_t count = static_cast<uint32_t>(remaining < ubLength_ ? remaining : ubLength_);
        const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
        auto& local = usePing ? ping : pong;

        // 同一块 UB 的写回完成后，才允许下一次搬入。
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        ProcessTile<Aligned>(local, offset, count, eventId);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        usePing = !usePing;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    // 每个完整 tile 已对齐，整段对齐时所有搬运都可使用 DataCopy。
    if (blockLength_ % (COPY_ALIGN_BYTES / sizeof(T)) == 0) {
        ProcessTiles<true>();
    } else {
        ProcessTiles<false>();
    }
    PipeBarrier<PIPE_ALL>();
}

} // namespace NsRelu
#endif // RELU_H
