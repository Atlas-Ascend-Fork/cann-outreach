// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BLOCK_SIZE = 256; // 每个 tile 256 个元素，32B 对齐

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length)
    {
        totalLength = length;
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ DT_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z, totalLength);
        pipe.InitBuffer(inQueueX, 1, BLOCK_SIZE * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, BLOCK_SIZE * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, BLOCK_SIZE * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 按固定 tile 轮转分配，避免总长度不能被核数整除时漏算。
        for (uint32_t offset = blockIdx * BLOCK_SIZE; offset < totalLength; offset += blockNum * BLOCK_SIZE) {
            uint32_t tileLength = totalLength - offset;
            if (tileLength > BLOCK_SIZE) tileLength = BLOCK_SIZE;

            AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
            AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
            AscendC::DataCopy(xLocal, xGm[offset], tileLength);
            AscendC::DataCopy(yLocal, yGm[offset], tileLength);
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);

            xLocal = inQueueX.DeQue<DT_X>();
            yLocal = inQueueY.DeQue<DT_X>();
            AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
            AscendC::Mul(zLocal, xLocal, yLocal, tileLength);
            outQueueZ.EnQue<DT_X>(zLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);

            zLocal = outQueueZ.DeQue<DT_X>();
            AscendC::DataCopy(zGm[offset], zLocal, tileLength);
            outQueueZ.FreeTensor(zLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t totalLength = 0;
    uint32_t blockIdx = 0;
    uint32_t blockNum = 1;
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
