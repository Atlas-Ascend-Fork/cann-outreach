// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2; // 双缓冲

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    
    // 补全参数，接收 tiling 数据
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData& tiling) {
        // 获取 Tiling 参数
        uint32_t totalLength = tiling.totalLength;
        this->tileNum = tiling.tileNum;
        
        // 计算每个核处理的数据长度（多核并行）
        this->blockLength = totalLength / AscendC::GetBlockNum();
        // 计算单次处理（搬运/计算）的元素数
        this->tileLength = this->blockLength / this->tileNum;
        
        // 设置每个核的 Global Memory 起始地址
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        // 为队列分配内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    
    __aicore__ inline void Process() {
        // 循环处理每个数据块
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
        
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        
        // 执行矢量乘法计算
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
        
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // 补齐私有成员变量
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    // 将获取到的 tiling_data 传入 Init
    op.Init(x, y, z, tiling_data);
    op.Process();
}