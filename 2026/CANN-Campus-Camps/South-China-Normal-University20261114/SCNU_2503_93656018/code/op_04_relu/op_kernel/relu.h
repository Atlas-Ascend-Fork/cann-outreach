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
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* data,
        TPipe* pipe)
    {
        const int64_t offset =
            GetBlockIdx() * data->blockFactor;

        const int64_t remaining =
            data->totalNum - offset;

        if (remaining <= 0) {
            return;
        }

        length_ = static_cast<uint32_t>(
            remaining < data->blockFactor
                ? remaining
                : data->blockFactor);

        tile_ = static_cast<uint32_t>(data->ubFactor);

        input_.SetGlobalBuffer(
            (__gm__ T*)x + offset,
            length_);

        output_.SetGlobalBuffer(
            (__gm__ T*)y + offset,
            length_);

        pipe->InitBuffer(
            ping_,
            tile_ * sizeof(T));

        if (length_ > tile_) {
            pipe->InitBuffer(
                pong_,
                tile_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (length_ == 0) {
            return;
        }

        const uint32_t tiles =
            (length_ + tile_ - 1) / tile_;

        // 预取前两块。
        Load(0, 0, Count(0));

        if (tiles > 1) {
            Load(1, tile_, Count(tile_));
        }

        for (
            uint32_t i = 0, offset = 0;
            i < tiles;
            ++i, offset += tile_)
        {
            const uint32_t slot = i & 1;

            const event_t event =
                static_cast<event_t>(slot);

            auto local = Buffer(slot);
            const uint32_t count = Count(offset);

            // 当前块搬入完成后才能计算。
            WaitFlag<HardEvent::MTE2_V>(event);

            // 输入输出完全重叠，原地执行 ReLU。
            AscendC::Relu(
                local,
                local,
                count);

            // 计算完成后才能搬出。
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);

            Store(local, offset, count);

            if (i + 2 < tiles) {
                // 同一块 UB 再次作为输入前，
                // 必须等待它上一次写回 GM 结束。
                SetFlag<HardEvent::MTE3_MTE2>(event);
                WaitFlag<HardEvent::MTE3_MTE2>(event);

                const uint32_t nextOffset =
                    offset + 2 * tile_;

                Load(
                    slot,
                    nextOffset,
                    Count(nextOffset));
            }
        }

        // 保证最后的搬出完成。
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline uint32_t Count(
        uint32_t offset)
    {
        const uint32_t remaining = length_ - offset;

        return remaining < tile_
            ? remaining
            : tile_;
    }

    __aicore__ inline LocalTensor<T> Buffer(
        uint32_t slot)
    {
        return slot == 0
            ? ping_.Get<T>()
            : pong_.Get<T>();
    }

    __aicore__ inline void Load(
        uint32_t slot,
        uint32_t offset,
        uint32_t count)
    {
        auto local = Buffer(slot);

        if (count % (32 / sizeof(T)) == 0) {
            DataCopy(
                local,
                input_[offset],
                count);
        } else {
            DataCopyExtParams params{
                1,
                static_cast<uint32_t>(
                    count * sizeof(T)),
                0,
                0,
                0
            };

            DataCopyPadExtParams<T> pad{
                false,
                0,
                0,
                0
            };

            DataCopyPad(
                local,
                input_[offset],
                params,
                pad);
        }

        SetFlag<HardEvent::MTE2_V>(
            static_cast<event_t>(slot));
    }

    __aicore__ inline void Store(
        LocalTensor<T> local,
        uint32_t offset,
        uint32_t count)
    {
        if (count % (32 / sizeof(T)) == 0) {
            DataCopy(
                output_[offset],
                local,
                count);
        } else {
            DataCopyExtParams params{
                1,
                static_cast<uint32_t>(
                    count * sizeof(T)),
                0,
                0,
                0
            };

            DataCopyPad(
                output_[offset],
                local,
                params);
        }
    }

    TBuf<QuePosition::VECIN> ping_;
    TBuf<QuePosition::VECIN> pong_;

    GlobalTensor<T> input_;
    GlobalTensor<T> output_;

    uint32_t length_ = 0;
    uint32_t tile_ = 0;
};

} // namespace NsRelu

#endif


