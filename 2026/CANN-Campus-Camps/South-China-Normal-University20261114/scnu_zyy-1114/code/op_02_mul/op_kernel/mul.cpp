// Kernel side implementation: z = x * y
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t QUEUE_DEPTH = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t TILE_DATA_NUM = 1024;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length)
    {
        constexpr uint32_t alignElems = BLOCK_SIZE / sizeof(DT_X);
        uint32_t totalBlocks = (length + alignElems - 1) / alignElems;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t blocksPerCore = totalBlocks / blockNum;
        uint32_t tailBlocks = totalBlocks % blockNum;
        uint32_t myBlocks = blocksPerCore + (blockIdx < tailBlocks ? 1 : 0);
        uint32_t myBlockStart = blockIdx * blocksPerCore
                              + (blockIdx < tailBlocks ? blockIdx : tailBlocks);
        uint32_t offset = myBlockStart * alignElems;

        this->coreDataNum = myBlocks * alignElems;
        if (offset >= length) {
            this->coreDataNum = 0;
        } else if (offset + this->coreDataNum > length) {
            this->coreDataNum = length - offset;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_DATA_NUM * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_DATA_NUM * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_DATA_NUM * sizeof(DT_X));

        this->tileNum = (this->coreDataNum + TILE_DATA_NUM - 1) / TILE_DATA_NUM;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; i++) {
            this->processDataNum = (i == this->tileNum - 1)
                ? (this->coreDataNum - i * TILE_DATA_NUM) : TILE_DATA_NUM;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_DATA_NUM], this->processDataNum);
        AscendC::DataCopy(yLocal, yGm[progress * TILE_DATA_NUM], this->processDataNum);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, this->processDataNum);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * TILE_DATA_NUM], zLocal, this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t processDataNum = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
