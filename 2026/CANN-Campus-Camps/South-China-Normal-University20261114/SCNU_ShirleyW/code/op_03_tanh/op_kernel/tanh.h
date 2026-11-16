#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y,
        const TanhTilingData* tilingData)
    {
        const int64_t offset =
            GetBlockIdx() * tilingData->blockFactor;

        const int64_t remaining =
            tilingData->totalNum - offset;

        blockLength_ =
            remaining > tilingData->blockFactor
                ? tilingData->blockFactor
                : remaining;

        if (blockLength_ < 0) {
            blockLength_ = 0;
        }

        ubLength_ = tilingData->ubFactor;

        inputGMX.SetGlobalBuffer(
            (__gm__ T*)x + offset, blockLength_);
        outputGMY.SetGlobalBuffer(
            (__gm__ T*)y + offset, blockLength_);

        pipe.InitBuffer(
            inputQueueX, BUFFER_NUM,
            ubLength_ * sizeof(T));
        pipe.InitBuffer(
            outputQueueY, BUFFER_NUM,
            ubLength_ * sizeof(T));

        pipe.InitBuffer(
            xFloatBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(
            aBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(
            bBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(
            resultBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(
            polyBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(
            maskBuf, ((ubLength_ + 255) / 256) * 32);
    }

    __aicore__ inline void Process()
    {
        for (int64_t start = 0, progress = 0;
             start < blockLength_;
             start += ubLength_, ++progress) {
            const int64_t remaining =
                blockLength_ - start;

            const int64_t currentNum =
                remaining < ubLength_
                    ? remaining : ubLength_;

            CopyIn(progress, currentNum);
            Compute(currentNum);
            CopyOut(progress, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(
        int64_t progress, int64_t currentNum)
    {
        LocalTensor<T> xLocal =
            inputQueueX.AllocTensor<T>();

        DataCopy(
            xLocal,
            inputGMX[progress * ubLength_],
            currentNum);

        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<T> xLocal =
            inputQueueX.DeQue<T>();
        LocalTensor<T> yLocal =
            outputQueueY.AllocTensor<T>();

        const uint32_t n =
            static_cast<uint32_t>(currentNum);

        LocalTensor<float> xf =
            xFloatBuf.Get<float>();
        LocalTensor<float> a =
            aBuf.Get<float>();
        LocalTensor<float> b =
            bBuf.Get<float>();
        LocalTensor<float> result =
            resultBuf.Get<float>();
        LocalTensor<float> poly =
            polyBuf.Get<float>();
        LocalTensor<uint8_t> mask =
            maskBuf.Get<uint8_t>();

        // float16 输入提升到 float32 计算。
        if constexpr (sizeof(T) == sizeof(half)) {
            Cast(xf, xLocal, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
        } else {
            xf = xLocal.template ReinterpretCast<float>();
        }

        // 限制指数输入，避免大数溢出。
        Mins(a, xf, 10.0f, n);
        PipeBarrier<PIPE_V>();

        Maxs(a, a, -10.0f, n);
        PipeBarrier<PIPE_V>();

        Muls(b, a, -1.0f, n);
        PipeBarrier<PIPE_V>();

        // a = exp(x)，b = exp(-x)。
        Exp(a, a, n);
        Exp(b, b, n);
        PipeBarrier<PIPE_V>();

        // result = (exp(x)-exp(-x)) /
        //          (exp(x)+exp(-x))。
        Sub(result, a, b, n);
        Add(poly, a, b, n);
        PipeBarrier<PIPE_V>();

        Div(result, result, poly, n);
        PipeBarrier<PIPE_V>();

        // 零附近使用 x-x^3/3，减少指数相减的误差。
        // 先限制多项式输入，避免未选中的分支溢出。
        Mins(a, xf, 0.01f, n);
        PipeBarrier<PIPE_V>();

        Maxs(a, a, -0.01f, n);
        PipeBarrier<PIPE_V>();

        Mul(b, a, a, n);
        PipeBarrier<PIPE_V>();

        Muls(b, b, -1.0f / 3.0f, n);
        PipeBarrier<PIPE_V>();

        Adds(b, b, 1.0f, n);
        PipeBarrier<PIPE_V>();

        Mul(poly, a, b, n);
        Abs(a, xf, n);
        PipeBarrier<PIPE_V>();

        CompareScalar(
            mask, a, 0.01f, CMPMODE::LT, n);
        PipeBarrier<PIPE_V>();

        Select(
            b, mask, poly, result,
            SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();

        if constexpr (sizeof(T) == sizeof(half)) {
            Cast(yLocal, b, RoundMode::CAST_RINT, n);
        } else {
            Adds(yLocal, b, 0.0f, n);
        }

        PipeBarrier<PIPE_V>();

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(
        int64_t progress, int64_t currentNum)
    {
        LocalTensor<T> yLocal =
            outputQueueY.DeQue<T>();

        DataCopy(
            outputGMY[progress * ubLength_],
            yLocal,
            currentNum);

        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    TBuf<QuePosition::VECCALC> xFloatBuf;
    TBuf<QuePosition::VECCALC> aBuf;
    TBuf<QuePosition::VECCALC> bBuf;
    TBuf<QuePosition::VECCALC> resultBuf;
    TBuf<QuePosition::VECCALC> polyBuf;
    TBuf<QuePosition::VECCALC> maskBuf;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

} // namespace NsTanh

#endif // TANH_H