#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"
template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              GM_ADDR workspace, GM_ADDR tiling)
{
    constexpr uint32_t TOTAL_LENGTH = 8 * 2048;
    const uint32_t count = TOTAL_LENGTH / AscendC::GetBlockNum();
    const uint32_t offset = AscendC::GetBlockIdx() * count;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, count);
    yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, count);
    zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, count);

    const uint32_t strideBytes = count * sizeof(DT_X) + (count < 16384 ? 32 : 0);
    AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECCALC, 0, count);
    AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECCALC, strideBytes, count);
    AscendC::LocalTensor<DT_X> zLocal(AscendC::TPosition::VECCALC, 2 * strideBytes, count);

    AscendC::DataCopy(xLocal, xGm, count);
    AscendC::DataCopy(yLocal, yGm, count);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Mul(zLocal, xLocal, yLocal, count);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(zGm, zLocal, count);
}
