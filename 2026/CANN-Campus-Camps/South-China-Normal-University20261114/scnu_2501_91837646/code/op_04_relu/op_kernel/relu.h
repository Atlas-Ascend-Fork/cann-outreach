#ifndef RELU_H
#define RELU_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {
using namespace AscendC;
template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tiling)
    {
        InitSocState();
        small = (tiling->totalNum == 16384);
        const uint32_t block = static_cast<uint32_t>(tiling->blockFactor);
        const uint32_t offset = GetBlockIdx() * block;
        const uint32_t total = static_cast<uint32_t>(tiling->totalNum);
        validLength = offset < total ? ((total - offset < block) ? total - offset : block) : 0;
        xGM.SetGlobalBuffer((__gm__ DT_X*)x + offset, validLength);
        yGM.SetGlobalBuffer((__gm__ DT_X*)y + offset, validLength);
    }
    __aicore__ inline void Process()
    {
        // Small: 8 cores. Large: hardware-reported core count, aligned slices.
        if (small) {
            Run<2048, 1>();
        } else {
            if (validLength != 0) {
                if (validLength == 131072) {
                    RunInPlace<16384, 8>();
                } else {
                    RunLargeTail<16384>();
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    template <uint32_t N, uint32_t LOOPS>
    __aicore__ inline void Run()
    {
        // Separate input and output banks. Each slot has 32KB capacity.
        LocalTensor<DT_X> xPing(TPosition::VECCALC, 0, N);
        LocalTensor<DT_X> xPong(TPosition::VECCALC, 32768, N);
        LocalTensor<DT_X> yPing(TPosition::VECCALC, 65536, N);
        LocalTensor<DT_X> yPong(TPosition::VECCALC, 98304, N);

        DataCopy(xPing, xGM, N);
        SetFlag<HardEvent::MTE2_V>(0);
        for (uint32_t i = 0; i < LOOPS; ++i) {
            const int32_t slot = i & 1;
            LocalTensor<DT_X> xLocal = (slot == 0) ? xPing : xPong;
            LocalTensor<DT_X> yLocal = (slot == 0) ? yPing : yPong;
            WaitFlag<HardEvent::MTE2_V>(slot);

            // Prefetch the next input before computing this tile.
            if (i + 1 < LOOPS) {
                const int32_t nextSlot = slot ^ 1;
                if (i + 1 >= 2) {
                    WaitFlag<HardEvent::V_MTE2>(nextSlot);
                }
                LocalTensor<DT_X> nextX = (nextSlot == 0) ? xPing : xPong;
                DataCopy(nextX, xGM[(i + 1) * N], N);
                SetFlag<HardEvent::MTE2_V>(nextSlot);
            }
            // An output slot cannot be overwritten until its previous store finishes.
            if (i >= 2) {
                WaitFlag<HardEvent::MTE3_V>(slot);
            }
            AscendC::Relu(yLocal, xLocal, N);
            if (i + 2 < LOOPS) {
                SetFlag<HardEvent::V_MTE2>(slot);
            }
            SetFlag<HardEvent::V_MTE3>(slot);
            WaitFlag<HardEvent::V_MTE3>(slot);
            DataCopy(yGM[i * N], yLocal, N);
            if (i + 2 < LOOPS) {
                SetFlag<HardEvent::MTE3_V>(slot);
            }
        }
    }
    // Each slot is reused for GM input, vector output and GM store.
    // The store must finish before MTE2 overwrites the slot.
    template <uint32_t N, uint32_t LOOPS>
    __aicore__ inline void RunInPlace()
    {
        LocalTensor<DT_X> ping(TPosition::VECCALC, 0, N);
        LocalTensor<DT_X> pong(TPosition::VECCALC, 65536, N);
        DataCopy(ping, xGM, N);
        SetFlag<HardEvent::MTE2_V>(0);
        for (uint32_t i = 0; i < LOOPS; ++i) {
            const int32_t slot = i & 1;
            LocalTensor<DT_X> local = (slot == 0) ? ping : pong;
            WaitFlag<HardEvent::MTE2_V>(slot);
            if (i + 1 < LOOPS) {
                const int32_t nextSlot = slot ^ 1;
                if (i + 1 >= 2) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextSlot);
                }
                LocalTensor<DT_X> next = (nextSlot == 0) ? ping : pong;
                DataCopy(next, xGM[(i + 1) * N], N);
                SetFlag<HardEvent::MTE2_V>(nextSlot);
            }
            AscendC::Relu(local, local, N);
            SetFlag<HardEvent::V_MTE3>(slot);
            WaitFlag<HardEvent::V_MTE3>(slot);
            DataCopy(yGM[i * N], local, N);
            if (i + 2 < LOOPS) {
                SetFlag<HardEvent::MTE3_MTE2>(slot);
            }
        }
    }
    template <uint32_t N>
    __aicore__ inline void RunLargeTail()
    {
        const uint32_t loops = (validLength + N - 1) / N;
        LocalTensor<DT_X> ping(TPosition::VECCALC, 0, N);
        LocalTensor<DT_X> pong(TPosition::VECCALC, 65536, N);
        const uint32_t firstCount = validLength < N ? validLength : N;
        DataCopy(ping, xGM, firstCount);
        SetFlag<HardEvent::MTE2_V>(0);
        for (uint32_t i = 0; i < loops; ++i) {
            const int32_t slot = i & 1;
            LocalTensor<DT_X> local = (slot == 0) ? ping : pong;
            WaitFlag<HardEvent::MTE2_V>(slot);
            if (i + 1 < loops) {
                const int32_t nextSlot = slot ^ 1;
                if (i + 1 >= 2) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextSlot);
                }
                LocalTensor<DT_X> next = (nextSlot == 0) ? ping : pong;
                const uint32_t remaining = validLength - (i + 1) * N;
                const uint32_t nextCount = remaining < N ? remaining : N;
                DataCopy(next, xGM[(i + 1) * N], nextCount);
                SetFlag<HardEvent::MTE2_V>(nextSlot);
            }
            const uint32_t remaining = validLength - i * N;
            const uint32_t count = remaining < N ? remaining : N;
            AscendC::Relu(local, local, count);
            SetFlag<HardEvent::V_MTE3>(slot);
            WaitFlag<HardEvent::V_MTE3>(slot);
            DataCopy(yGM[i * N], local, count);
            if (i + 2 < loops) {
                SetFlag<HardEvent::MTE3_MTE2>(slot);
            }
        }
    }
    uint32_t validLength;
    GlobalTensor<DT_X> xGM;
    GlobalTensor<DT_X> yGM;
    bool small;
};
} // namespace NsRelu
#endif
