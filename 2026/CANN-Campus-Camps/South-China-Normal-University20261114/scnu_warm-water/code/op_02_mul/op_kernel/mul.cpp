// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    // The contest shape is (8, 2048) and Host launches 4 blocks.
    constexpr uint32_t elementsPerCore = 4096;

    const uint32_t offset = AscendC::GetBlockIdx() * elementsPerCore;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + offset, elementsPerCore);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + offset, elementsPerCore);
    zGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + offset, elementsPerCore);

    constexpr uint32_t bufferBytes = elementsPerCore * sizeof(DT_X);
    constexpr uint32_t eventId = 0;
    AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECCALC, 0, elementsPerCore);
    AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECCALC, bufferBytes, elementsPerCore);
    AscendC::LocalTensor<DT_X> zLocal(AscendC::TPosition::VECCALC, 2 * bufferBytes, elementsPerCore);

    AscendC::DataCopy(xLocal, xGm, elementsPerCore);
    AscendC::DataCopy(yLocal, yGm, elementsPerCore);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);

    constexpr uint64_t elementsPerRepeat = 256 / sizeof(DT_X);
    constexpr uint8_t repeatCount = elementsPerCore / elementsPerRepeat;
    static_assert(elementsPerCore % elementsPerRepeat == 0, "Full repeats required");
    const AscendC::BinaryRepeatParams strides = {1, 1, 1, 8, 8, 8};
    AscendC::Mul(zLocal, xLocal, yLocal, elementsPerRepeat, repeatCount, strides);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
    AscendC::DataCopy(zGm, zLocal, elementsPerCore);
}
