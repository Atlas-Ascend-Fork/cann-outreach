// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

// 每个AI Core处理一个连续分块：16384 / 8 = 2048个元素。
template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->tileLength = tileLength;

        const uint32_t blockOffset = GetBlockIdx() * tileLength;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + blockOffset, tileLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + blockOffset, tileLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + blockOffset, tileLength);

        pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm, tileLength);
        DataCopy(yLocal, yGm, tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        Mul(zLocal, xLocal, yLocal, tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm, zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

    uint32_t totalLength = 0;
    uint32_t tileLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
