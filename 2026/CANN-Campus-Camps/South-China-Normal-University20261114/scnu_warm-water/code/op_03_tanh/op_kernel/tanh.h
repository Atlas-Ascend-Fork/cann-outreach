/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData*)
    {
        AscendC::InitSocState();
        const uint32_t offset = AscendC::GetBlockIdx() * tileLength;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + offset, tileLength);
        output.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + offset, tileLength);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t ioBytes = tileLength * sizeof(DT_X);
        constexpr uint32_t floatBytes = tileLength * sizeof(float);
        LocalTensor<DT_X> xLocal(TPosition::VECCALC, 0, tileLength);
        LocalTensor<DT_X> yLocal(TPosition::VECCALC, ioBytes, tileLength);
        LocalTensor<float> work(TPosition::VECCALC, 2 * ioBytes, tileLength);
        LocalTensor<float> pos(TPosition::VECCALC, 2 * ioBytes + floatBytes, tileLength);
        LocalTensor<float> neg(TPosition::VECCALC, 2 * ioBytes + 2 * floatBytes, tileLength);

        DataCopy(xLocal, input, tileLength);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        // The lower bound keeps exp(-2*x) from overflowing.
        if constexpr (std::is_same<DT_X, half>::value) {
            Cast(work, xLocal, RoundMode::CAST_NONE, tileLength);
            PipeBarrier<PIPE_V>();
            Maxs(work, work, -10.0f, tileLength);
        } else {
            Maxs(work, xLocal, -10.0f, tileLength);
        }
        PipeBarrier<PIPE_V>();

        // tanh(x) = (1 - exp(-2*x)) / (1 + exp(-2*x)).
        Muls(pos, work, -2.0f, tileLength);
        PipeBarrier<PIPE_V>();
        Exp(pos, pos, tileLength);
        PipeBarrier<PIPE_V>();
        Muls(neg, pos, -1.0f, tileLength);
        Adds(work, pos, 1.0f, tileLength);
        PipeBarrier<PIPE_V>();
        Adds(neg, neg, 1.0f, tileLength);
        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<DT_X, half>::value) {
            Div(pos, neg, work, tileLength);
            PipeBarrier<PIPE_V>();
            Cast(yLocal, pos, RoundMode::CAST_RINT, tileLength);
        } else {
            Div(yLocal, neg, work, tileLength);
        }

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(output, yLocal, tileLength);
    }

private:
    static constexpr uint32_t tileLength = 2048;
    GlobalTensor<DT_X> input;
    GlobalTensor<DT_X> output;
};

} // namespace NsTanh
#endif // TANH_H
