#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z,
        uint32_t totalLength, uint32_t tileNum)
    {
        blockLength = totalLength / AscendC::GetBlockNum();
        loopCount = tileNum * BUFFER_NUM;
        tileLength = blockLength / loopCount;

        const uint32_t offset =
            blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, blockLength);

        pipe.InitBuffer(
            inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(
            inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(
            outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t i)
    {
        auto x = inQueueX.AllocTensor<DT_X>();
        auto y = inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(x, xGm[i * tileLength], tileLength);
        AscendC::DataCopy(y, yGm[i * tileLength], tileLength);

        inQueueX.EnQue(x);
        inQueueY.EnQue(y);
    }

    __aicore__ inline void Compute()
    {
        auto x = inQueueX.DeQue<DT_X>();
        auto y = inQueueY.DeQue<DT_X>();
        auto z = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(z, x, y, tileLength);

        outQueueZ.EnQue(z);
        inQueueX.FreeTensor(x);
        inQueueY.FreeTensor(y);
    }

    __aicore__ inline void CopyOut(uint32_t i)
    {
        auto z = outQueueZ.DeQue<DT_X>();

        AscendC::DataCopy(zGm[i * tileLength], z, tileLength);

        outQueueZ.FreeTensor(z);
    }

    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t loopCount;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}