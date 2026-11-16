
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 256;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength)
    {
        blockLength =
            totalLength / AscendC::GetBlockNum();

        tileNum = blockLength / TILE_LENGTH;

        uint32_t offset =
            blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset, blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset, blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X*)z + offset, blockLength);

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(
            xLocal,
            xGm[progress * TILE_LENGTH],
            TILE_LENGTH);

        AscendC::DataCopy(
            yLocal,
            yGm[progress * TILE_LENGTH],
            TILE_LENGTH);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            TILE_LENGTH);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        AscendC::DataCopy(
            zGm[progress * TILE_LENGTH],
            zLocal,
            TILE_LENGTH);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> inQueueX;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM> inQueueY;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.length);

    op.Process();
}

