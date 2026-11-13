#ifndef RELU_H
#define RELU_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"
namespace NsRelu {
using namespace AscendC;
template <typename T, uint32_t N>
__aicore__ inline void SmallRelu(GM_ADDR x, GM_ADDR y)
{
    LocalTensor<T> data(TPosition::VECCALC, 0, N);
    GlobalTensor<T> in;
    GlobalTensor<T> out;
    in.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * N, N);
    out.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * N, N);
    DataCopy(data, in, N);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Relu(data, data, N);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(out, data, N);
    PipeBarrier<PIPE_ALL>();
}
template <typename T>
__aicore__ inline void LargeRelu(GM_ADDR x, GM_ADDR y)
{
    constexpr uint32_t TILE = 65536 / sizeof(T);
    constexpr uint32_t BLOCK = 131072;
    constexpr uint32_t LOOPS = BLOCK / TILE;
    LocalTensor<T> buffers(TPosition::VECCALC, 0, 2 * TILE);
    GlobalTensor<T> in;
    GlobalTensor<T> out;
    in.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * BLOCK, BLOCK);
    out.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * BLOCK, BLOCK);
    DataCopy(buffers, in, TILE);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
#pragma unroll
    for (uint32_t i = 0; i < LOOPS; ++i) {
        const auto id = static_cast<event_t>(i & 1);
        if (i + 1 < LOOPS) {
            const auto nextId = static_cast<event_t>((i + 1) & 1);
            if (i >= 1) WaitFlag<HardEvent::MTE3_MTE2>(nextId);
            DataCopy(buffers[((i + 1) & 1) * TILE], in[(i + 1) * TILE], TILE);
            SetFlag<HardEvent::MTE2_V>(nextId);
        }
        auto data = buffers[(i & 1) * TILE];
        WaitFlag<HardEvent::MTE2_V>(id);
        AscendC::Relu(data, data, TILE);
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
        DataCopy(out[i * TILE], data, TILE);
        if (i + 2 < LOOPS) SetFlag<HardEvent::MTE3_MTE2>(id);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
class Relu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* td)
    { x_ = x; y_ = y; td_ = td; }
    __aicore__ inline void Process()
    {
        if (td_->totalNum == 16384 && td_->blockFactor == 1024) {
            SmallRelu<T, 1024>(x_, y_);
            return;
        }
        const int64_t start = GetBlockIdx() * td_->blockFactor;
        if (start >= td_->totalNum) return;
        const int64_t length = Min(td_->blockFactor, td_->totalNum - start);
        const int64_t tile = td_->ubFactor;
        LocalTensor<T> buffers(TPosition::VECCALC, 0, 2 * tile);
        GlobalTensor<T> in;
        GlobalTensor<T> out;
        in.SetGlobalBuffer((__gm__ T*)x_ + start, length);
        out.SetGlobalBuffer((__gm__ T*)y_ + start, length);
        const int64_t loops = (length + tile - 1) / tile;
        CopyIn(buffers, in, Min(tile, length));
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (int64_t i = 0; i < loops; ++i) {
            const auto id = static_cast<event_t>(i & 1);
            if (i + 1 < loops) {
                const auto nextId = static_cast<event_t>((i + 1) & 1);
                if (i >= 1) WaitFlag<HardEvent::MTE3_MTE2>(nextId);
                const int64_t nextPos = (i + 1) * tile;
                CopyIn(buffers[((i + 1) & 1) * tile], in[nextPos], Min(tile, length - nextPos));
                SetFlag<HardEvent::MTE2_V>(nextId);
            }
            auto data = buffers[(i & 1) * tile];
            const int64_t pos = i * tile;
            const uint32_t n = Min(tile, length - pos);
            WaitFlag<HardEvent::MTE2_V>(id);
            AscendC::Relu(data, data, n);
            SetFlag<HardEvent::V_MTE3>(id);
            WaitFlag<HardEvent::V_MTE3>(id);
            if (n * sizeof(T) % 32 == 0) {
                DataCopy(out[pos], data, n);
            } else {
                DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
                DataCopyPad(out[pos], data, cp);
            }
            if (i + 2 < loops) SetFlag<HardEvent::MTE3_MTE2>(id);
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    __aicore__ inline void CopyIn(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t n)
    {
        if (n * sizeof(T) % 32 == 0) {
            DataCopy(dst, src, n);
        } else {
            DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(dst, src, cp, pad);
        }
    }
    __aicore__ inline int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
    GM_ADDR x_;
    GM_ADDR y_;
    const ReluTilingData* td_;
};
}
#endif
