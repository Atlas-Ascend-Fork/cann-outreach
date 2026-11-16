/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 *
 * 实现 y = max(0, x)：
 *   1. 多核并行：按 blockFactor 把展平后的一维数据均分到各 AIV 核；
 *   2. UB 切分：每核再按 ubFactor 分块；
 *   3. 手写 ping-pong 流水：一次 InitBuffer 拿到 2*BUFFER_NUM 块 UB，
 *      用显式 SetFlag/WaitFlag 组织“预取下一块 + 计算当前块 + 写回当前块”，
 *      使 MTE2 / Vector / MTE3 并行；
 *   4. 对齐快路径：整块 32B 对齐时用 DataCopy，只有尾部残块才用 DataCopyPad。
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;


constexpr int32_t BUFFER_NUM = 2;


constexpr int64_t GM_ALIGN_BYTES = 32;


constexpr int32_t BYTES_PER_REPEAT = 256;


constexpr int32_t MAX_REPEAT_TIMES = 255;

constexpr event_t EV_IN_MTE2_V[BUFFER_NUM] = {EVENT_ID0, EVENT_ID2};
constexpr event_t EV_IN_V_MTE2[BUFFER_NUM] = {EVENT_ID1, EVENT_ID3};
constexpr event_t EV_OUT_V_MTE3[BUFFER_NUM] = {EVENT_ID4, EVENT_ID6};
constexpr event_t EV_OUT_MTE3_V[BUFFER_NUM] = {EVENT_ID5, EVENT_ID7};

template <typename T>
class Relu {
public:
    __aicore__ inline explicit Relu(TPipe* pipe) : pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t TileLength(int64_t progress) const;
    __aicore__ inline bool IsAligned(int64_t currentNum) const;
    __aicore__ inline LocalTensor<T> InTensor(int64_t slot) const;
    __aicore__ inline LocalTensor<T> OutTensor(int64_t slot) const;
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe* pipe_ = nullptr;

    TBuf<TPosition::VECIN> ubBuf_;
    LocalTensor<T> ubLocal_;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t slotElems_ = 0;
    int64_t tileNum_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t gmOffset = blockIdx * blockLength_;
    const int64_t remainNum = tilingData->totalNum - gmOffset;
    if (remainNum < blockLength_) {
        blockLength_ = remainNum;
    }
    if (blockLength_ <= 0) {
        blockLength_ = 0;
        return;
    }
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    slotElems_ =
        ((ubLength_ * static_cast<int64_t>(sizeof(T)) + GM_ALIGN_BYTES - 1) / GM_ALIGN_BYTES) *
        GM_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
    tileNum_ = (blockLength_ + ubLength_ - 1) / ubLength_;

    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x) + gmOffset, static_cast<uint32_t>(blockLength_));
    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) + gmOffset, static_cast<uint32_t>(blockLength_));

    pipe_->InitBuffer(
        ubBuf_, static_cast<uint32_t>(slotElems_ * sizeof(T) * BUFFER_NUM * 2));
    ubLocal_ = ubBuf_.Get<T>();
}

template <typename T>
__aicore__ inline LocalTensor<T> Relu<T>::InTensor(int64_t slot) const
{
    return ubLocal_[static_cast<int32_t>(slot * slotElems_)];
}

template <typename T>
__aicore__ inline LocalTensor<T> Relu<T>::OutTensor(int64_t slot) const
{
    return ubLocal_[static_cast<int32_t>((BUFFER_NUM + slot) * slotElems_)];
}

template <typename T>
__aicore__ inline int64_t Relu<T>::TileLength(int64_t progress) const
{
    const int64_t restNum = blockLength_ - progress * ubLength_;
    return (restNum < ubLength_) ? restNum : ubLength_;
}

template <typename T>
__aicore__ inline bool Relu<T>::IsAligned(int64_t currentNum) const
{
    return ((currentNum * static_cast<int64_t>(sizeof(T))) % GM_ALIGN_BYTES) == 0;
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);

    if (progress >= BUFFER_NUM) {
        WaitFlag<HardEvent::V_MTE2>(EV_IN_V_MTE2[slot]);
    }

    LocalTensor<T> xLocal = InTensor(slot);
    const int64_t offset = progress * ubLength_;
    if (IsAligned(currentNum)) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EV_IN_MTE2_V[slot]);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t progress, int64_t currentNum)
{
    const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);
    WaitFlag<HardEvent::MTE2_V>(EV_IN_MTE2_V[slot]);

    if (progress >= BUFFER_NUM) {
        WaitFlag<HardEvent::MTE3_V>(EV_OUT_MTE3_V[slot]);
    }

    LocalTensor<T> xLocal = InTensor(slot);
    LocalTensor<T> yLocal = OutTensor(slot);

    constexpr int32_t maxNumPerCall =
        MAX_REPEAT_TIMES * (BYTES_PER_REPEAT / static_cast<int32_t>(sizeof(T)));
    int64_t doneNum = 0;
    while (doneNum < currentNum) {
        const int64_t restNum = currentNum - doneNum;
        const int32_t callNum = (restNum < static_cast<int64_t>(maxNumPerCall))
                                    ? static_cast<int32_t>(restNum)
                                    : maxNumPerCall;
        Maxs(yLocal[static_cast<int32_t>(doneNum)],
             xLocal[static_cast<int32_t>(doneNum)],
             static_cast<T>(0),
             callNum);
        doneNum += callNum;
    }

    SetFlag<HardEvent::V_MTE3>(EV_OUT_V_MTE3[slot]);
    if (progress + BUFFER_NUM < tileNum_) {
        SetFlag<HardEvent::V_MTE2>(EV_IN_V_MTE2[slot]);
    }
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);
    WaitFlag<HardEvent::V_MTE3>(EV_OUT_V_MTE3[slot]);

    LocalTensor<T> yLocal = OutTensor(slot);
    const int64_t offset = progress * ubLength_;
    if (IsAligned(currentNum)) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }

    if (progress + BUFFER_NUM < tileNum_) {
        SetFlag<HardEvent::MTE3_V>(EV_OUT_MTE3_V[slot]);
    }
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    CopyIn(0, TileLength(0));
    for (int64_t i = 0; i < tileNum_; ++i) {
        const int64_t currentNum = TileLength(i);
        if (i + 1 < tileNum_) {
            CopyIn(i + 1, TileLength(i + 1));
        }
        Compute(i, currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H