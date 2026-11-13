#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

// Valid for the fixed 16384-element contract enforced by the accompanying Host.
// The Host launches 1, 2, 4, or 8 AIV cores.
template <typename DT_X, uint32_t BLOCK_LENGTH>
__aicore__ inline void RunMulStatic(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    static_assert(sizeof(DT_X) == 2 || sizeof(DT_X) == 4,
                  "Only float16 and float32 are supported");
    constexpr uint32_t BUFFER_BYTES = BLOCK_LENGTH * sizeof(DT_X);
    static_assert(BUFFER_BYTES % 32 == 0, "DataCopy requires 32-byte alignment");

    const uint32_t offset = AscendC::GetBlockIdx() * BLOCK_LENGTH;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, BLOCK_LENGTH);
    yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, BLOCK_LENGTH);
    zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, BLOCK_LENGTH);

    // Byte offsets in this core's UB; no TPipe/TBuf is used.
    // Three disjoint ranges whose total size matches the Host UB check.
    const AscendC::LocalTensor<DT_X> xLocal(
        AscendC::TPosition::VECCALC, 0, BLOCK_LENGTH);
    const AscendC::LocalTensor<DT_X> yLocal(
        AscendC::TPosition::VECCALC, BUFFER_BYTES, BLOCK_LENGTH);
    const AscendC::LocalTensor<DT_X> zLocal(
        AscendC::TPosition::VECCALC, 2 * BUFFER_BYTES, BLOCK_LENGTH);

    AscendC::DataCopy(xLocal, xGm, BLOCK_LENGTH);
    AscendC::DataCopy(yLocal, yGm, BLOCK_LENGTH);

    // Static-Tensor programming owns its event IDs. The directions are distinct.
    // Event ID 0 is available here; reserved IDs 6 and 7 are not used.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

    AscendC::Mul(zLocal, xLocal, yLocal, BLOCK_LENGTH);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(zGm, zLocal, BLOCK_LENGTH);
}

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);

    // Required without TPipe: initialize mask and atomic-operation state.
    AscendC::InitSocState();

    // The Host already validates length and selects a supported core count.
    // Avoid loading the redundant blockLength from GM; each length is constant.
    const uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum == 4) {
        RunMulStatic<DT_X, 4096>(x, y, z);
    } else if (blockNum == 8) {
        RunMulStatic<DT_X, 2048>(x, y, z);
    } else if (blockNum == 2) {
        RunMulStatic<DT_X, 8192>(x, y, z);
    } else if (blockNum == 1) {
        RunMulStatic<DT_X, 16384>(x, y, z);
    }

    // Static-Tensor kernels finish all pipeline work before returning.
    AscendC::PipeBarrier<PIPE_ALL>();
}
