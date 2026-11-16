#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

// Fixed workload: (8, 2048), one tile per core; no runtime tiling loads.
template <typename DT_X>
class KernelMul {
public:
    static constexpr uint32_t CORES = sizeof(DT_X) == 2 ? MUL_FP16_CORES : MUL_FP32_CORES;
    static constexpr uint32_t LENGTH = MUL_TOTAL_LENGTH / CORES;
    static constexpr uint32_t BYTES = LENGTH * sizeof(DT_X);
    static constexpr uint32_t LANES = 256 / sizeof(DT_X);
    // Atlas A2 / ascend910b UB layout experiment.
    // Separate bank regions; shift the second source by half a 512B group cycle.
    static constexpr uint32_t X_ADDR = 0;
    static constexpr uint32_t Y_ADDR = 64 * 1024 + 256;
    static constexpr uint32_t Z_ADDR = 128 * 1024;
    static_assert(X_ADDR + BYTES <= Y_ADDR, "x/y overlap");
    static_assert(Y_ADDR + BYTES <= Z_ADDR, "y/z overlap");
    static_assert(Z_ADDR + BYTES <= 192 * 1024, "UB capacity exceeded");

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        const uint32_t offset = AscendC::GetBlockIdx() * LENGTH;
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, LENGTH);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, LENGTH);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, LENGTH);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECCALC, X_ADDR, LENGTH);
        AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECCALC, Y_ADDR, LENGTH);
        AscendC::LocalTensor<DT_X> zLocal(AscendC::TPosition::VECCALC, Z_ADDR, LENGTH);
        AscendC::DataCopy(xLocal, xGm, LENGTH);
        AscendC::DataCopy(yLocal, yGm, LENGTH);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Mul<DT_X>(zLocal, xLocal, yLocal,
                          static_cast<uint64_t>(LANES),
                          static_cast<uint8_t>(LENGTH / LANES),
                          AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(zGm, zLocal, LENGTH);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(MulTilingData);
    // Launch geometry is selected on the host; no GM tiling read is needed.
    KernelMul<DT_X> op;
    op.Init(x, y, z);
    op.Process();
}



