// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        length_ = length;
        xGm.SetGlobalBuffer((__gm__ DT_X*)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z, length);
        pipe.InitBuffer(xQueue, 1, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(yQueue, 1, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(zQueue, 1, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        const uint32_t stride = AscendC::GetBlockNum() * TILE_LENGTH;
        for (uint32_t offset = AscendC::GetBlockIdx() * TILE_LENGTH;
             offset < length_; offset += stride) {
            auto xLocal = xQueue.AllocTensor<DT_X>();
            auto yLocal = yQueue.AllocTensor<DT_X>();
            AscendC::DataCopy(xLocal, xGm[offset], TILE_LENGTH);
            AscendC::DataCopy(yLocal, yGm[offset], TILE_LENGTH);
            xQueue.EnQue(xLocal);
            yQueue.EnQue(yLocal);

            xLocal = xQueue.DeQue<DT_X>();
            yLocal = yQueue.DeQue<DT_X>();
            auto zLocal = zQueue.AllocTensor<DT_X>();
            AscendC::Mul(zLocal, xLocal, yLocal, TILE_LENGTH);
            zQueue.EnQue(zLocal);
            xQueue.FreeTensor(xLocal);
            yQueue.FreeTensor(yLocal);

            zLocal = zQueue.DeQue<DT_X>();
            AscendC::DataCopy(zGm[offset], zLocal, TILE_LENGTH);
            zQueue.FreeTensor(zLocal);
        }
    }

private:
    // The problem specifies eight rows of 2048 elements.
    static constexpr uint32_t TILE_LENGTH = 2048;
    uint32_t length_;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue, yQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQueue;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
