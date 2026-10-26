/*!
 * \file exp.h
 * \brief Exp with static UB storage and explicit pipeline synchronization.
 */
#ifndef EXP_H
#define EXP_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {
using namespace AscendC;

// No TPipe/TQue: event 0 in each event type belongs to this kernel.
template <typename T>
__aicore__ inline void RunTile(const GlobalTensor<T>& xGm, const GlobalTensor<T>& yGm,
    const LocalTensor<T>& xLocal, const LocalTensor<T>& yLocal, uint32_t count)
{
    const bool aligned = (count * sizeof(T)) % 32 == 0;
    if (aligned) {
        DataCopy(xLocal, xGm, count);
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm, params, padding);
    }
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    AscendC::Exp(yLocal, xLocal, count);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    if (aligned) {
        DataCopy(yGm, yLocal, count);
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm, yLocal, params);
    }
}

template <typename T>
class Exp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tiling)
    {
        if (tiling->blockFactor <= 0 || tiling->ubFactor <= 0) {
            return;
        }
        const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tiling->blockFactor;
        if (offset >= tiling->totalNum) {
            return;
        }
        const int64_t remaining = tiling->totalNum - offset;
        length_ = remaining < tiling->blockFactor ? remaining : tiling->blockFactor;
        tile_ = static_cast<uint32_t>(tiling->ubFactor);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, length_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, length_);
    }

    __aicore__ inline void Process()
    {
        if (length_ == 0) {
            return;
        }
        // Host aligns tile_ to 32B and budgets four tiles; only two are used.
        LocalTensor<T> xLocal(TPosition::VECIN, 0, tile_);
        LocalTensor<T> yLocal(TPosition::VECOUT, tile_ * sizeof(T), tile_);
        int64_t progress = 0;
        while (progress < length_) {
            const int64_t remaining = length_ - progress;
            const uint32_t count = remaining < tile_ ? static_cast<uint32_t>(remaining) : tile_;
            RunTile(xGm_[progress], yGm_[progress], xLocal, yLocal, count);
            progress += count;
            if (progress < length_) {
                // Finish the previous copy-out before reusing UB storage.
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t length_ = 0;
    uint32_t tile_ = 0;
};

template <typename T>
__aicore__ inline void RunFixed2048(GM_ADDR x, GM_ADDR y)
{
    constexpr uint32_t count = 2048;
    const uint32_t block = GetBlockIdx();
    if (block >= 8) {
        return;
    }
    const uint32_t offset = block * count;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, count);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, count);
    LocalTensor<T> xLocal(TPosition::VECIN, 0, count);
    // Shift the output start by one 256B vector span; test this UB layout
    // against v8's 512B-congruent input/output starts on the real device.
    constexpr uint32_t outputGapBytes = 256;
    LocalTensor<T> yLocal(TPosition::VECOUT, count * sizeof(T) + outputGapBytes, count);
    // One DataCopy call per direction, described as eight adjacent bursts.
    // DataCopyParams::blockLen is measured in 32-byte DataBlocks.
    constexpr uint16_t bursts = 8;
    static_assert((count * sizeof(T)) % (bursts * 32) == 0);
    const DataCopyParams transfer{bursts,
        static_cast<uint16_t>(count * sizeof(T) / (bursts * 32)), 0, 0};
    DataCopy(xLocal, xGm, transfer);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    AscendC::Exp(yLocal, xLocal, count);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(yGm, yLocal, transfer);
}

template <typename T>
__aicore__ inline void RunExp(GM_ADDR x, GM_ADDR y, const ExpTilingData* tiling)
{
    if (tiling->totalNum == 16384 && tiling->blockFactor == 2048 && tiling->ubFactor >= 2048) {
        RunFixed2048<T>(x, y);
        return;
    }
    Exp<T> op;
    op.Init(x, y, tiling);
    op.Process();
}
} // namespace NsExp
#endif
