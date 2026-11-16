// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t MUL_TILE_LENGTH = 256;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        tileCount = length / MUL_TILE_LENGTH;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z, length);

        pipe.InitBuffer(inQueueX, 1, MUL_TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, MUL_TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, MUL_TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        // 每个核从自己的编号开始，按核数跨块处理。
        uint32_t coreCount = AscendC::GetBlockNum();
        for (uint32_t tile = AscendC::GetBlockIdx(); tile < tileCount; tile += coreCount) {
            CopyIn(tile);
            Compute();
            CopyOut(tile);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tile) {
        uint32_t offset = tile * MUL_TILE_LENGTH;
        auto xLocal = inQueueX.AllocTensor<DT_X>();
        auto yLocal = inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(xLocal, xGm[offset], MUL_TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[offset], MUL_TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute() {
        auto xLocal = inQueueX.DeQue<DT_X>();
        auto yLocal = inQueueY.DeQue<DT_X>();
        auto zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, MUL_TILE_LENGTH);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tile) {
        uint32_t offset = tile * MUL_TILE_LENGTH;
        auto zLocal = outQueueZ.DeQue<DT_X>();

        AscendC::DataCopy(zGm[offset], zLocal, MUL_TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t tileCount;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}