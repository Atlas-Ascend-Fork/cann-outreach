// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData &tilingData) {
        const uint32_t blockIndex = AscendC::GetBlockIdx();
        const uint32_t extraBefore = blockIndex < tilingData.extraCores ? blockIndex : tilingData.extraCores;
        const uint32_t offset = blockIndex * tilingData.baseLength + extraBefore * MUL_GROUP_ELEMENTS;
        length = tilingData.baseLength + (blockIndex < tilingData.extraCores ? MUL_GROUP_ELEMENTS : 0);

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + offset, length);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + offset, length);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(z) + offset, length);
    }

    __aicore__ inline void Process() {
        // 少核时也将三个缓冲区的总容量限制在 48 KiB 以内。
        constexpr uint32_t MAX_TILE_LENGTH = 4096;
        const uint32_t tileLength = length < MAX_TILE_LENGTH ? length : MAX_TILE_LENGTH;
        const uint32_t tileBytes = tileLength * sizeof(DT_X);
        AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECCALC, 0, tileLength);
        AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECCALC, tileBytes, tileLength);
        AscendC::LocalTensor<DT_X> zLocal(AscendC::TPosition::VECCALC, 2 * tileBytes, tileLength);

        for (uint32_t offset = 0; offset < length; offset += tileLength) {
            const uint32_t remaining = length - offset;
            const uint32_t count = remaining < tileLength ? remaining : tileLength;
            AscendC::DataCopy(xLocal, xGm[offset], count);
            AscendC::DataCopy(yLocal, yGm[offset], count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::Mul(zLocal, xLocal, yLocal, count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(zGm[offset], zLocal, count);

            // 写回完成后才复用 UB，也保证最后一批写回完成。
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t length;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    AscendC::InitSocState();
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}
