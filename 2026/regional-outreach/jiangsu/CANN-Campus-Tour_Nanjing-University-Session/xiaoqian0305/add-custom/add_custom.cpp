#include "kernel_operator.h"

using namespace AscendC;

constexpr int BUFFER_NUM = 2; // double buffering

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileLength)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = this->blockLength / tileLength + (this->blockLength % tileLength != 0);
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ half*)x + GetBlockIdx() * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + GetBlockIdx() * this->blockLength, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + GetBlockIdx() * this->blockLength, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        Add(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;

    uint32_t blockLength; // length of data each core processes
    uint32_t tileNum;     // number of tiles per core
    uint32_t tileLength;  // length per tile
};

// tiling data structure, passed from host via tiling buffer
struct AddCustomTilingData {
    uint32_t totalLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    AddCustomTilingData tilingData;
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
