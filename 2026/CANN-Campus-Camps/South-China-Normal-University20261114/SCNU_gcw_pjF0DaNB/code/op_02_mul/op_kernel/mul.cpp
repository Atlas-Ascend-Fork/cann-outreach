// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 const MulTilingData* tilingData)
    {
        uint32_t totalLength = tilingData->length;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockLength = (totalLength + blockNum - 1) / blockNum;
        uint32_t blockOffset = blockIdx * blockLength;

        if (blockOffset + blockLength > totalLength) {
            blockLength = (blockOffset < totalLength) ? (totalLength - blockOffset) : 0;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + blockOffset, blockLength);

        constexpr uint32_t TILE_NUM = 2;
        uint32_t tileLength = (blockLength + TILE_NUM - 1) / TILE_NUM;
        uint32_t alignNum = 32 / sizeof(DT_X);
        tileLength = (tileLength + alignNum - 1) / alignNum * alignNum;
        this->tileLength = tileLength;
        this->coreLength = blockLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = (coreLength + tileLength - 1) / tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t offset = i * tileLength;
            uint32_t len = (offset + tileLength > coreLength)
                           ? (coreLength - offset) : tileLength;
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::DataCopy(yLocal, yGm[offset], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, len);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t coreLength;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, &tilingData);
    op.Process();
}
