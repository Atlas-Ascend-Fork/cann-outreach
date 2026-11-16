// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class T>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        uint32_t alignElem = 32 / sizeof(T);
        uint32_t totalBlocks = length / alignElem;
        
        uint32_t coreNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        
        if (blockIdx >= coreNum || blockIdx >= totalBlocks) {
            this->blockLength = 0;
            return;
        }
        
        uint32_t blocksPerCore = totalBlocks / coreNum;
        uint32_t remBlocks = totalBlocks % coreNum;
        
        uint32_t myBlocks = blocksPerCore + (blockIdx < remBlocks ? 1 : 0);
        uint32_t myOffsetBlocks = blockIdx * blocksPerCore + (blockIdx < remBlocks ? blockIdx : remBlocks);
        
        this->blockLength = myBlocks * alignElem;
        uint32_t offset = myOffsetBlocks * alignElem;
        
        if (this->blockLength == 0) return;

        xGm.SetGlobalBuffer((__gm__ T*)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T*)z + offset, this->blockLength);

        this->tileLength = this->blockLength / BUFFER_NUM;
        this->tileLength = this->tileLength / alignElem * alignElem;
        if (this->tileLength == 0) this->tileLength = alignElem;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
    }
    
    __aicore__ inline void Process() {
        if (this->blockLength == 0) return;
        
        // 计算能被完整处理的循环次数，以及最后剩下的尾部长度
        int32_t loopCount = this->blockLength / this->tileLength;
        int32_t tailLength = this->blockLength % this->tileLength;

        // 1. 处理完整对齐的主块
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        
        // 2. 处理尾部残余数据（这就是之前丢掉的 1.56%）
        if (tailLength > 0) {
            CopyIn(loopCount, tailLength);
            Compute(loopCount, tailLength);
            CopyOut(loopCount, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t curLen)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        
        DataCopy(xLocal, xGm[progress * this->tileLength], curLen);
        DataCopy(yLocal, yGm[progress * this->tileLength], curLen);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress, uint32_t curLen)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        
        Mul(zLocal, xLocal, yLocal, curLen);
        
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress, uint32_t curLen)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(zGm[progress * this->tileLength], zLocal, curLen);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<T> xGm, yGm, zGm;
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