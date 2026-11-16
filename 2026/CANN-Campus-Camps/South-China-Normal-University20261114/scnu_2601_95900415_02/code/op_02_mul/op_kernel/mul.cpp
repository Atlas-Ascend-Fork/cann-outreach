#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t blockLength) {
        this->blockLength = blockLength;
        uint32_t offset = blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, blockLength);

        // 仅使用两个队列，复用 inQueueX 作为输出队列，减少一次同步信号开销
        pipe.InitBuffer(inQueueX, 1, blockLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, blockLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        // 1. CopyIn
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm, this->blockLength);
        AscendC::DataCopy(yLocal, yGm, this->blockLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);

        // 2. Compute (In-place 原地计算直接覆盖 xLocal)
        xLocal = inQueueX.DeQue<DT_X>();
        yLocal = inQueueY.DeQue<DT_X>();
        AscendC::Mul(xLocal, xLocal, yLocal, this->blockLength);
        inQueueY.FreeTensor(yLocal);

        // 3. CopyOut (直接从 xLocal 写回 GM，无需再 EnQue/DeQue)
        // 注意：需要插入MTE2/V到MTE3的流水线同步，最安全且零多余队列的写法如下：
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(zGm, xLocal, this->blockLength);
        inQueueX.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.blockLength);
    op.Process();
}