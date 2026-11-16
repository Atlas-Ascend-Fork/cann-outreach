#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);

    // 固定输入 (8, 2048)，Host 启动 8 个矢量核。
    constexpr uint32_t LENGTH = 2048;
    constexpr uint32_t BYTES = LENGTH * sizeof(DT_X);

    const uint32_t offset =
        AscendC::GetBlockIdx() * LENGTH;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    xGm.SetGlobalBuffer(
        (__gm__ DT_X*)x + offset, LENGTH);
    yGm.SetGlobalBuffer(
        (__gm__ DT_X*)y + offset, LENGTH);
    zGm.SetGlobalBuffer(
        (__gm__ DT_X*)z + offset, LENGTH);

    // 紧凑 UB 布局，无 TPipe 和队列。
    AscendC::LocalTensor<DT_X> xLocal(
        AscendC::TPosition::VECCALC,
        0,
        LENGTH);

    AscendC::LocalTensor<DT_X> yLocal(
        AscendC::TPosition::VECCALC,
        BYTES,
        LENGTH);

    AscendC::DataCopy(xLocal, xGm, LENGTH);
    AscendC::DataCopy(yLocal, yGm, LENGTH);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
        EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
        EVENT_ID0);

    // 原地计算，省去独立输出缓冲区。
    AscendC::Mul(xLocal, xLocal, yLocal, LENGTH);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
        EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
        EVENT_ID0);

    AscendC::DataCopy(zGm, xLocal, LENGTH);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
        EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
        EVENT_ID0);
}