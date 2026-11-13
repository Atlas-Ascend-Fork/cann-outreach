
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length)
    {
        (void)length;

        
        // float32: 4核，每核4096个元素
        constexpr uint32_t BLOCK_LENGTH = 4096;

        const uint32_t offset =
            AscendC::GetBlockIdx() * BLOCK_LENGTH;

        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset,
            BLOCK_LENGTH);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset,
            BLOCK_LENGTH);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X*)z + offset,
            BLOCK_LENGTH);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t BLOCK_LENGTH = 4096;

        constexpr uint32_t BUFFER_BYTES =
            BLOCK_LENGTH * sizeof(DT_X);

        // 静态Tensor
        AscendC::LocalTensor<DT_X> xLocal(
            AscendC::TPosition::VECCALC,
            0,
            BLOCK_LENGTH);

        AscendC::LocalTensor<DT_X> yLocal(
            AscendC::TPosition::VECCALC,
            BUFFER_BYTES,
            BLOCK_LENGTH);

        AscendC::LocalTensor<DT_X> zLocal(
            AscendC::TPosition::VECCALC,
            2 * BUFFER_BYTES,
            BLOCK_LENGTH);

        // CopyIn
        AscendC::DataCopy(
            xLocal, xGm, BLOCK_LENGTH);

        AscendC::DataCopy(
            yLocal, yGm, BLOCK_LENGTH);

        AscendC::SetFlag<
            AscendC::HardEvent::MTE2_V>(0);

        AscendC::WaitFlag<
            AscendC::HardEvent::MTE2_V>(0);

        // Compute
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            BLOCK_LENGTH);

        AscendC::SetFlag<
            AscendC::HardEvent::V_MTE3>(0);

        AscendC::WaitFlag<
            AscendC::HardEvent::V_MTE3>(0);

        // CopyOut
        AscendC::DataCopy(
            zGm, zLocal, BLOCK_LENGTH);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    AscendC::InitSocState();

    REGISTER_TILING_DEFAULT(MulTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.length);

    op.Process();
}
