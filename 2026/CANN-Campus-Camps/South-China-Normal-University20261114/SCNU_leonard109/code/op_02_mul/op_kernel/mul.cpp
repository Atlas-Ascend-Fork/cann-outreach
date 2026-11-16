// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        totalLength = length;
        xGm.SetGlobalBuffer((__gm__ DT_X*)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z, length);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, MUL_TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, MUL_TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, MUL_TILE_LENGTH * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        const uint32_t tileCount = totalLength / MUL_TILE_LENGTH +
                                   (totalLength % MUL_TILE_LENGTH != 0);
        for (uint32_t tile = AscendC::GetBlockIdx(); tile < tileCount;
             tile += AscendC::GetBlockNum()) {
            const uint32_t offset = tile * MUL_TILE_LENGTH;
            const uint32_t remaining = totalLength - offset;
            const uint32_t count = remaining < MUL_TILE_LENGTH ? remaining : MUL_TILE_LENGTH;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        if (count == MUL_TILE_LENGTH) {
            AscendC::DataCopy(xLocal, xGm[offset], count);
            AscendC::DataCopy(yLocal, yGm[offset], count);
        } else {
            // Copy only valid bytes; pad the local buffer to a 32-byte boundary.
            const uint32_t alignElements = 32 / sizeof(DT_X);
            const uint8_t rightPadding = static_cast<uint8_t>(
                (alignElements - count % alignElements) % alignElements);
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padding{true, 0, rightPadding, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[offset], params, padding);
            AscendC::DataCopyPad(yLocal, yGm[offset], params, padding);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, count);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        if (count == MUL_TILE_LENGTH) {
            AscendC::DataCopy(zGm[offset], zLocal, count);
        } else {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(zGm[offset], zLocal, params);
        }
        outQueueZ.FreeTensor(zLocal);
    }

    static constexpr int32_t BUFFER_NUM = 2;
    uint32_t totalLength = 0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
