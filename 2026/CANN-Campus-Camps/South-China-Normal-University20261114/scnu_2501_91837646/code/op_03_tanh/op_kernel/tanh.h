/* Fixed-shape Tanh: 16384 elements, 8 cores, 2048 elements/core. */
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
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               const TanhTilingData* tilingData)
    {
        // Required for static-tensor programming without a TPipe object.
        InitSocState();
        // Matches host blockFactor=ubFactor=2048, blockDim=8.
        const uint32_t offset = GetBlockIdx() * LENGTH;
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, LENGTH);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, LENGTH);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<DT_X> xLocal(TPosition::VECCALC, 0, LENGTH);
        // Input has been consumed before the final result is written.
        // Reuse its buffer for output; GM input/output remain separate.
        LocalTensor<DT_X> yLocal = xLocal;
        // Separate read/write banks; offset sum by 256 bytes so the
        // two Div source operands occupy disjoint bank-group halves.
        LocalTensor<float> pos(TPosition::VECCALC, 65536, LENGTH);
        LocalTensor<float> sum(TPosition::VECCALC, 131328, LENGTH);

        DataCopy(xLocal, inputGMX, LENGTH);
        const int32_t inEvent = 0;
        SetFlag<HardEvent::MTE2_V>(inEvent);
        WaitFlag<HardEvent::MTE2_V>(inEvent);

        // Same arithmetic as the measured v6 implementation.
        if constexpr (std::is_same<DT_X, float>::value) {
            Mins(pos, xLocal, 10.0f, LENGTH);
        } else {
            Cast(sum, xLocal, RoundMode::CAST_NONE, LENGTH);
            Mins(pos, sum, 10.0f, LENGTH);
        }
        Muls(pos, pos, 2.0f, LENGTH);
        Exp(pos, pos, LENGTH);
        Adds(sum, pos, 1.0f, LENGTH);
        Adds(pos, pos, -1.0f, LENGTH);
        if constexpr (std::is_same<DT_X, float>::value) {
            Div(yLocal, pos, sum, LENGTH);
        } else {
            Div(pos, pos, sum, LENGTH);
            Cast(yLocal, pos, RoundMode::CAST_RINT, LENGTH);
        }

        const int32_t outEvent = 0;
        SetFlag<HardEvent::V_MTE3>(outEvent);
        WaitFlag<HardEvent::V_MTE3>(outEvent);
        DataCopy(outputGMY, yLocal, LENGTH);

        // No TPipe destructor exists here: explicitly complete all pipelines.
        PipeBarrier<PIPE_ALL>();

    }

private:
    static constexpr uint32_t LENGTH = 2048;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
};
} // namespace NsTanh
#endif
