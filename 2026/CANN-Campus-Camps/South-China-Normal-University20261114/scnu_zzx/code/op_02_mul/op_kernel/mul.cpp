#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length
    )
    {
        blockLength = length / AscendC::GetBlockNum();
        const uint32_t offset = blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, blockLength);

        pipe.InitBuffer(
            inQueueX,
            1,
            TILE_LENGTH * sizeof(DT_X)
        );
        pipe.InitBuffer(
            inQueueY,
            1,
            TILE_LENGTH * sizeof(DT_X)
        );
        pipe.InitBuffer(
            outQueueZ,
            1,
            TILE_LENGTH * sizeof(DT_X)
        );
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength; offset += TILE_LENGTH) {
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
        AscendC::DataCopy(xLocal, xGm[offset], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[offset], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 鎸夊厓绱犵浉涔橈紝杈撳叆鍜岃緭鍑轰娇鐢ㄧ浉鍚岀殑鏁版嵁绫诲瀷銆?
        AscendC::Mul(zLocal, xLocal, yLocal, TILE_LENGTH);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

    static constexpr uint32_t TILE_LENGTH = 256;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling
)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}