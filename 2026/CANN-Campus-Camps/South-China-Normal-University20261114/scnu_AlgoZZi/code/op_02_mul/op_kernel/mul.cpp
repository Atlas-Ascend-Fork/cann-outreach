// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t MUL_TILE_LENGTH = 1024;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        totalLength = length;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z, length);
        pipe.InitBuffer(inQueueX, 1, MUL_TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, MUL_TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, MUL_TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        uint64_t stride = (uint64_t)AscendC::GetBlockNum() * MUL_TILE_LENGTH;
        for (uint64_t offset = (uint64_t)AscendC::GetBlockIdx() * MUL_TILE_LENGTH;
             offset < totalLength; offset += stride) {
            uint32_t count = totalLength - offset < MUL_TILE_LENGTH
                ? (uint32_t)(totalLength - offset) : MUL_TILE_LENGTH;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.template AllocTensor<DT_X>();
        if (count * sizeof(DT_X) % 32 == 0) {
            AscendC::DataCopy(xLocal, xGm[offset], count);
            AscendC::DataCopy(yLocal, yGm[offset], count);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
            AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.template AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, count);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.template DeQue<DT_X>();
        if (count * sizeof(DT_X) % 32 == 0) {
            AscendC::DataCopy(zGm[offset], zLocal, count);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        }
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t totalLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
