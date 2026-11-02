#ifndef EXP_PIPELINE_H
#define EXP_PIPELINE_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {
using namespace AscendC;
template <typename DT_X>
class Exp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData*)
    {
        const uint32_t offset = blockLength * GetBlockIdx();
        inputGM.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        outputGM.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);
    }
    __aicore__ inline void Process()
    {
        auto x0 = Input(0); auto x1 = Input(1);
        auto y0 = Output(0); auto y1 = Output(1);
        Load(x0, 0); Load(x1, 1);
        ComputeStore(x0, y0, 0);
        if constexpr (tileCount == 2) {
            ComputeStore(x1, y1, 1);
        } else {
            auto x2 = Input(2); auto x3 = Input(3);
            auto y2 = Output(2); auto y3 = Output(3);
            Load(x2, 2);
            ComputeStore(x1, y1, 1);
            Load(x3, 3);
            ComputeStore(x2, y2, 2);
            ComputeStore(x3, y3, 3);
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    static constexpr uint32_t blockLength = 2048;
    static constexpr uint32_t tileCount = 2;
    static constexpr uint32_t tileLength = blockLength / tileCount;
    GlobalTensor<DT_X> inputGM, outputGM;
    __aicore__ inline LocalTensor<DT_X> Input(uint32_t tile)
    {
        return LocalTensor<DT_X>(TPosition::VECIN, tile * tileLength * sizeof(DT_X), tileLength);
    }
    __aicore__ inline LocalTensor<DT_X> Output(uint32_t tile)
    {
        return LocalTensor<DT_X>(TPosition::VECOUT, (blockLength + tile * tileLength) * sizeof(DT_X), tileLength);
    }
    __aicore__ inline void Load(LocalTensor<DT_X>& local, uint32_t tile)
    {
        DataCopy(local, inputGM[tile * tileLength], tileLength);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(tile));
    }
    __aicore__ inline void ComputeStore(LocalTensor<DT_X>& x, LocalTensor<DT_X>& y, uint32_t tile)
    {
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(tile));
        constexpr uint64_t mask = 256U / sizeof(DT_X);
        constexpr uint8_t repeats = tileLength / mask;
        AscendC::Exp(y, x, mask, repeats, UnaryRepeatParams{1, 1, 8, 8});
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(tile));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(tile));
        DataCopy(outputGM[tile * tileLength], y, tileLength);
    }
};
}
#endif
