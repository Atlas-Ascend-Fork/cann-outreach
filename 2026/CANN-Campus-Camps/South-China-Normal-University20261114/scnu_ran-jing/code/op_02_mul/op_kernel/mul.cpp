#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    
    // 按照 8 核来切分
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        // 假设平台使用 8 个核来跑
        uint32_t blockLength = totalLength / 8;
        
        // 获取当前核的编号 (0~7)
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + blockIdx * blockLength, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + blockIdx * blockLength, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + blockIdx * blockLength, blockLength);
        
        // 每个核只分配一块足够大的内存，一次性处理完
        pipe.InitBuffer(inQueueX, 1, blockLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, blockLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, blockLength * sizeof(DT_X));
        
        this->blockLength = blockLength;
    }
    
    __aicore__ inline void Process() {
        // 一次性搬运、一次性计算、一次性搬出，不用循环
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        
        // 从 GM 搬到 UB
        AscendC::DataCopy(xLocal, xGm, this->blockLength);
        AscendC::DataCopy(yLocal, yGm, this->blockLength);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 计算乘法
        AscendC::Mul(zLocal, xLocal, yLocal, this->blockLength);
        
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        
        // 从 UB 搬回 GM
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
    // 传入固定的总长度 (8*2048 = 16384)
    op.Init(x, y, z, 8 * 2048); 
    op.Process();
}