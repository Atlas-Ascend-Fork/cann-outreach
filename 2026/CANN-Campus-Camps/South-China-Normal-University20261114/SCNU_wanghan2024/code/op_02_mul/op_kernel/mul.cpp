// Kernel侧核函数实现
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        this->blockLength = length; // 让 0 号核处理全部数据
        uint32_t offset = 0;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, blockLength);

        // 每次只搬运 256 个元素，防止一次性塞满 UB 导致 507035 错误
        this->tileLength = 256; 
        
        pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, tileLength * sizeof(DT_X));
    }
    
    __aicore__ inline void Process() {
        // 强制只用 0 号核，其他核直接退出，防多核冲突
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        // 计算需要循环多少次
        uint32_t loopCount = blockLength / tileLength;

        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t currentOffset = i * tileLength;
            CopyIn(currentOffset);
            Compute();
            CopyOut(currentOffset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xIn = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yIn = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 直接算乘法
        AscendC::Mul(zLocal, xIn, yIn, tileLength);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xIn);
        inQueueY.FreeTensor(yIn);
    }

    __aicore__ inline void CopyOut(uint32_t offset) {
        AscendC::LocalTensor<DT_X> zOut = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zOut, tileLength);
        outQueueZ.FreeTensor(zOut);
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
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}