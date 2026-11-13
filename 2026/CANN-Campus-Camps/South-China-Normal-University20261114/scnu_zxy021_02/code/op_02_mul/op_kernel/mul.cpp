// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData& tiling) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        
        this->blockLength = tiling.totalLength / blockNum;
        this->tileNum = tiling.tileNum;
        this->tileLength = this->blockLength / this->tileNum;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + blockIdx * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + blockIdx * this->blockLength, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + blockIdx * this->blockLength, this->blockLength);

        pipe.InitBuffer(inQueueX, 2, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 2, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 2, this->tileLength * sizeof(DT_X));
    }
    
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * tileLength], tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, tileLength);

        outQueueZ.EnQue(zLocal);
        
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data); 
    op.Process();
}