// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // 多核并行：按核号对全局数据做连续分片
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->blockLength = length / blockNum;
        uint32_t offset = blockIdx * this->blockLength;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->blockLength);
        // 为输入输出队列分配片上 UB 内存（单 tile 覆盖单核全部数据）
        pipe.InitBuffer(inQueueX, 1, this->blockLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, this->blockLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, this->blockLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        // CopyIn：Global -> Local
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm, this->blockLength);
        AscendC::DataCopy(yLocal, yGm, this->blockLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);

        // Compute：z = x * y
        xLocal = inQueueX.DeQue<DT_X>();
        yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, this->blockLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);

        // CopyOut：Local -> Global
        zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm, zLocal, this->blockLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
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
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
