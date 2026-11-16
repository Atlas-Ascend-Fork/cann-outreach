// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        xGm.SetGlobalBuffer((__gm__ DT_X*)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z, length);
        totalLength = length;
        pipe.InitBuffer(xQueue, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(yQueue, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(zQueue, 1, tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (uint32_t offset = AscendC::GetBlockIdx() * tileLength;
             offset < totalLength; offset += AscendC::GetBlockNum() * tileLength) {
            auto xLocal = xQueue.AllocTensor<DT_X>();
            auto yLocal = yQueue.AllocTensor<DT_X>();
            AscendC::DataCopy(xLocal, xGm[offset], tileLength);
            AscendC::DataCopy(yLocal, yGm[offset], tileLength);
            xQueue.EnQue(xLocal);
            yQueue.EnQue(yLocal);

            xLocal = xQueue.DeQue<DT_X>();
            yLocal = yQueue.DeQue<DT_X>();
            auto zLocal = zQueue.AllocTensor<DT_X>();
            AscendC::Mul(zLocal, xLocal, yLocal, tileLength);
            zQueue.EnQue(zLocal);
            xQueue.FreeTensor(xLocal);
            yQueue.FreeTensor(yLocal);

            zLocal = zQueue.DeQue<DT_X>();
            AscendC::DataCopy(zGm[offset], zLocal, tileLength);
            zQueue.FreeTensor(zLocal);
        }
    }
private:
    static constexpr uint32_t tileLength = 2048;
    uint32_t totalLength;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue, yQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> zQueue;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
