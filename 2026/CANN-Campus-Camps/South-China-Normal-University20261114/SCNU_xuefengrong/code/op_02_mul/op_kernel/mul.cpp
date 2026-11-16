// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling) {
        GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
        
        this->totalLength = tiling_data.totalLength;
        this->blockDim = tiling_data.blockDim;
        this->tileLength = tiling_data.tileLength;
        
        // 获取当前核的索引
        this->blockIdx = AscendC::GetBlockIdx();
        
        // 计算当前核在全局数据上的地址偏移
        uint32_t startOffset = this->blockIdx * this->tileLength;
        
        // 绑定GlobalTensor，指定当前核负责的数据段
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + startOffset, this->tileLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + startOffset, this->tileLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + startOffset, this->tileLength);
        
        // 初始化流水线，分配UB(Unified Buffer)内存大小
        pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, this->tileLength * sizeof(DT_X));
    }
    
    __aicore__ inline void Process() {
        // 单块数据直接执行 搬入->计算->搬出
        CopyIn(0);
        Compute(0);
        CopyOut(0);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        
        // 将数据从GM搬到UB
        AscendC::DataCopy(xLocal, xGm, this->tileLength);
        AscendC::DataCopy(yLocal, yGm, this->tileLength);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        
        // 执行逐元素乘法
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
        
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        
        // 将结果从UB搬到GM
        AscendC::DataCopy(zGm, zLocal, this->tileLength);
        
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
    
    uint32_t totalLength;
    uint32_t blockDim;
    uint32_t tileLength;
    uint32_t blockIdx;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling); // 将原模板的...替换为tiling
    op.Process();
}