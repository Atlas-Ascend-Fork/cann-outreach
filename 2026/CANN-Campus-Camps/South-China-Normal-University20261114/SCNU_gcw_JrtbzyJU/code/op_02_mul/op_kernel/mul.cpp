#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t kBufferNum = 2;

template <typename DT_X>
class KernelMul {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               uint32_t totalLength, uint32_t tileNum)
    {
        const uint32_t blockLength = totalLength / AscendC::GetBlockNum();
        const uint32_t blockOffset = AscendC::GetBlockIdx() * blockLength;

        tileCount = tileNum * kBufferNum;
        tileLength = blockLength / tileCount;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + blockOffset, blockLength);

        const uint32_t tileBytes = tileLength * sizeof(DT_X);
        pipe.InitBuffer(inQueueX, kBufferNum, tileBytes);
        pipe.InitBuffer(inQueueY, kBufferNum, tileBytes);
        pipe.InitBuffer(outQueueZ, kBufferNum, tileBytes);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t offset = tile * tileLength;
            CopyIn(offset);
            Compute();
            CopyOut(offset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset)
    {
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

    __aicore__ inline void CopyOut(uint32_t offset)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kBufferNum> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, kBufferNum> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kBufferNum> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t tileCount;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}