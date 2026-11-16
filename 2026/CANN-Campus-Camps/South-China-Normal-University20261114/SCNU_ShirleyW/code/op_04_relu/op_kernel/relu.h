#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y,
        const ReluTilingData* data)
    {
        small_ = data->totalNum <= 16384;

        const int64_t offset =
            GetBlockIdx() * data->blockFactor;

        const int64_t remaining =
            data->totalNum - offset;

        blockLength_ =
            remaining < data->blockFactor
                ? remaining : data->blockFactor;

        if (blockLength_ < 0) {
            blockLength_ = 0;
        }

        ubLength_ = data->ubFactor;

        inputGMX.SetGlobalBuffer(
            (__gm__ T*)x + offset, blockLength_);
        outputGMY.SetGlobalBuffer(
            (__gm__ T*)y + offset, blockLength_);

        if (small_) {
            pipe.InitBuffer(
                xBuffer, ubLength_ * sizeof(T));
            pipe.InitBuffer(
                yBuffer, ubLength_ * sizeof(T));
        } else {
            pipe.InitBuffer(
                inputQueueX, BUFFER_NUM,
                ubLength_ * sizeof(T));
            pipe.InitBuffer(
                outputQueueY, BUFFER_NUM,
                ubLength_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }

        if (small_) {
            ProcessSmall();
            return;
        }

        const int64_t loops =
            (blockLength_ + ubLength_ - 1) / ubLength_;

        CopyIn(0, TileCount(0));

        for (int64_t i = 0; i < loops; ++i) {
            // 第二块输入缓冲区预取下一块数据。
            if (i + 1 < loops) {
                CopyIn(i + 1, TileCount(i + 1));
            }

            const int64_t count = TileCount(i);
            Compute(count);
            CopyOut(i, count);
        }
    }

private:
    __aicore__ inline void ProcessSmall()
    {
        LocalTensor<T> x = xBuffer.Get<T>();
        LocalTensor<T> y = yBuffer.Get<T>();

        DataCopy(x, inputGMX, blockLength_);

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        AscendC::Relu(
            y, x, static_cast<int32_t>(blockLength_));

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopy(outputGMY, y, blockLength_);

        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }

    __aicore__ inline int64_t TileCount(
        int64_t progress)
    {
        const int64_t remaining =
            blockLength_ - progress * ubLength_;

        return remaining < ubLength_
            ? remaining : ubLength_;
    }

    __aicore__ inline void CopyIn(
        int64_t progress, int64_t count)
    {
        LocalTensor<T> x =
            inputQueueX.AllocTensor<T>();

        DataCopy(
            x,
            inputGMX[progress * ubLength_],
            count);

        inputQueueX.EnQue(x);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        LocalTensor<T> x =
            inputQueueX.DeQue<T>();
        LocalTensor<T> y =
            outputQueueY.AllocTensor<T>();

        AscendC::Relu(
            y, x, static_cast<int32_t>(count));

        outputQueueY.EnQue(y);
        inputQueueX.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(
        int64_t progress, int64_t count)
    {
        LocalTensor<T> y =
            outputQueueY.DeQue<T>();

        DataCopy(
            outputGMY[progress * ubLength_],
            y,
            count);

        outputQueueY.FreeTensor(y);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    TBuf<QuePosition::VECCALC> xBuffer;
    TBuf<QuePosition::VECCALC> yBuffer;

    bool small_ = false;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

} // namespace NsRelu

#endif // RELU_H