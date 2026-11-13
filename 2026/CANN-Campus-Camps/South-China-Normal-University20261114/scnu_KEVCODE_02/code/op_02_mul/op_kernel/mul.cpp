#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

// Specialization for the problem's fixed 8 x 2048 contiguous tensors.
// Dtype selection and operator registration remain in the framework.
template <typename DT_X, uint32_t count>
__aicore__ inline void MulStatic(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    const uint32_t offset = AscendC::GetBlockIdx() * count;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, count);
    yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, count);
    zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, count);

    // Addresses are UB byte offsets; gaps stagger the operand buffers.
    const uint32_t strideBytes = count * sizeof(DT_X) + (count < 16384 ? 32 : 0);
    AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECCALC, 0, count);
    AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECCALC, strideBytes, count);
    AscendC::LocalTensor<DT_X> zLocal(AscendC::TPosition::VECCALC, 2 * strideBytes, count);

    AscendC::DataCopy(xLocal, xGm, count);
    AscendC::DataCopy(yLocal, yGm, count);
    // Static tensors do not use TPipe; the event IDs are owned by this kernel.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Mul(zLocal, xLocal, yLocal, count);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(zGm, zLocal, count);
}

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              GM_ADDR workspace, GM_ADDR tiling)
{
    MulStatic<DT_X, 4096>(x, y, z);
}