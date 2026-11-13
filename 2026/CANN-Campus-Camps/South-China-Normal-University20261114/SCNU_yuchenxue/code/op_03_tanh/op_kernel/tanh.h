#ifndef TANH_H
#define TANH_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        this->blockLength =
            tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;
        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        const uint32_t offset =
            this->blockLength * AscendC::GetBlockIdx();

        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset, this->blockLength);
        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset, this->blockLength);

        pipe.InitBuffer(
            inputQueueX, BUFFER_NUM,
            this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(
            outputQueueY, BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        // 半精度输入也使用 float32 中间值。
        pipe.InitBuffer(
            floatInput, this->tileLength * sizeof(float));
        pipe.InitBuffer(
            exponent, this->tileLength * sizeof(float));
        pipe.InitBuffer(
            numerator, this->tileLength * sizeof(float));
        pipe.InitBuffer(
            denominator, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.AllocTensor<DT_X>();

        DataCopy(
            xLocal,
            inputGMX[progress * this->tileLength],
            this->tileLength);

        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        auto x = inputQueueX.DeQue<DT_X>();
        auto y = outputQueueY.AllocTensor<DT_X>();

        auto xf = floatInput.Get<float>();
        auto e = exponent.Get<float>();
        auto num = numerator.Get<float>();
        auto den = denominator.Get<float>();

        if constexpr (std::is_same<DT_X, half>::value) {
            Cast(xf, x, RoundMode::CAST_NONE, this->tileLength);
        } else {
            Adds(xf, x, 0.0f, this->tileLength);
        }

        PipeBarrier<PIPE_V>();

        // e = exp(-2 * abs(x))
        Abs(e, xf, this->tileLength);
        PipeBarrier<PIPE_V>();

        Muls(e, e, -2.0f, this->tileLength);
        PipeBarrier<PIPE_V>();

        Exp(e, e, this->tileLength);
        PipeBarrier<PIPE_V>();

        // magnitude = (1 - e) / (1 + e)
        Adds(den, e, 1.0f, this->tileLength);
        Muls(num, e, -1.0f, this->tileLength);
        PipeBarrier<PIPE_V>();

        Adds(num, num, 1.0f, this->tileLength);
        PipeBarrier<PIPE_V>();

        Div(num, num, den, this->tileLength);
        PipeBarrier<PIPE_V>();

        // 恢复符号，并避免 x=0 时除以零。
        Abs(den, xf, this->tileLength);
        PipeBarrier<PIPE_V>();

        Maxs(den, den, 1.0e-12f, this->tileLength);
        PipeBarrier<PIPE_V>();

        Div(xf, xf, den, this->tileLength);
        PipeBarrier<PIPE_V>();

        Mul(xf, xf, num, this->tileLength);
        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<DT_X, half>::value) {
            Cast(y, xf, RoundMode::CAST_RINT, this->tileLength);
        } else {
            Adds(y, xf, 0.0f, this->tileLength);
        }

        outputQueueY.EnQue(y);
        inputQueueX.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outputQueueY.DeQue<DT_X>();

        DataCopy(
            outputGMY[progress * this->tileLength],
            yLocal,
            this->tileLength);

        outputQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<TPosition::VECCALC> floatInput;
    TBuf<TPosition::VECCALC> exponent;
    TBuf<TPosition::VECCALC> numerator;
    TBuf<TPosition::VECCALC> denominator;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

}
#endif