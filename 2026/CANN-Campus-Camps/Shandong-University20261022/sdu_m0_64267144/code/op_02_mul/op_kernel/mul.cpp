#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z,
        uint32_t length, uint32_t tileLength, uint32_t tileCount)
    {
        this->tileLength = tileLength;
        this->tileCount = tileCount;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z, length);

        pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = AscendC::GetBlockIdx();
        const uint32_t coreCount = AscendC::GetBlockNum();
        for (uint32_t tile = coreId; tile < tileCount; tile += coreCount) {
            CopyIn(tile);
            Compute();
            CopyOut(tile);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tile)
    {
        const uint32_t offset = tile * tileLength;
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, tileLength);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tile)
    {
        const uint32_t offset = tile * tileLength;
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t tileLength;
    uint32_t tileCount;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData.length, tilingData.tileLength, tilingData.tileCount);
    op.Process();
}