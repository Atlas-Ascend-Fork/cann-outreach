#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData, tiling_data, tiling);

    const uint32_t length =
        tiling_data.length / AscendC::GetBlockNum();
    const uint32_t offset =
        AscendC::GetBlockIdx() * length;

    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, length);
    yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, length);
    zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, length);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC>
        xBuf, yBuf, zBuf;

    pipe.InitBuffer(xBuf, length * sizeof(DT_X));
    pipe.InitBuffer(yBuf, length * sizeof(DT_X));
    pipe.InitBuffer(zBuf, length * sizeof(DT_X));

    AscendC::LocalTensor<DT_X> xLocal = xBuf.Get<DT_X>();
    AscendC::LocalTensor<DT_X> yLocal = yBuf.Get<DT_X>();
    AscendC::LocalTensor<DT_X> zLocal = zBuf.Get<DT_X>();

    // 搬入两个输入，等待搬运完成后再计算。
    AscendC::DataCopy(xLocal, xGm, length);
    AscendC::DataCopy(yLocal, yGm, length);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

    // 逐元素乘法。
    AscendC::Mul(zLocal, xLocal, yLocal, length);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

    // 搬出结果，并在释放缓冲区前等待写回完成。
    AscendC::DataCopy(zGm, zLocal, length);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}