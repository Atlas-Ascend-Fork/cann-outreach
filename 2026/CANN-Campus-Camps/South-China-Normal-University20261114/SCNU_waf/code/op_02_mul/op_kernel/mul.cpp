// Kernel侧核函数实现
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        this->blockNum = GetBlockNum();
        this->blockIdx = GetBlockIdx();
        
        uint32_t normalBlockLength = length / this->blockNum;
        this->blockLength = normalBlockLength;
        if (this->blockIdx == this->blockNum - 1) {
            this->blockLength = length - this->blockIdx * normalBlockLength;
        }

        uint32_t blockOffset = this->blockIdx * normalBlockLength;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + blockOffset, this->blockLength);

        // 每次搬运 2048 个元素，分配 8192 字节的 buffer（足以应对 FP32，且防越界）
        pipe.InitBuffer(inQueueX, 2, 8192);
        pipe.InitBuffer(inQueueY, 2, 8192);
        pipe.InitBuffer(outQueueZ, 2, 8192);

        if (sizeof(DT_X) == 2) { // FP16 计算防溢出
            pipe.InitBuffer(tmpBufferX, 16384);
            pipe.InitBuffer(tmpBufferY, 16384);
        }
    }

    __aicore__ inline void Process() {
        uint32_t offset = 0;
        while (offset < this->blockLength) {
            uint32_t curLength = this->blockLength - offset;
            if (curLength > 2048) curLength = 2048; // 分块大小 2048
            
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
            
            offset += curLength;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], curLength);
        DataCopy(yLocal, yGm[offset], curLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        if (sizeof(DT_X) == 2) {
            LocalTensor<float> xFp32 = tmpBufferX.Get<float>();
            LocalTensor<float> yFp32 = tmpBufferY.Get<float>();
            
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, curLength);
            Cast(yFp32, yLocal, RoundMode::CAST_NONE, curLength);
            Mul(xFp32, xFp32, yFp32, curLength);
            Cast(zLocal, xFp32, RoundMode::CAST_RINT, curLength);
        } else {
            Mul(zLocal, xLocal, yLocal, curLength);
        }

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        outQueueZ.EnQue(zLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm[offset], zLocal, curLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX;
    TQue<QuePosition::VECIN, 2> inQueueY;
    TQue<QuePosition::VECOUT, 2> outQueueZ;
    TBuf<TPosition::VECCALC> tmpBufferX;
    TBuf<TPosition::VECCALC> tmpBufferY;
    GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t blockNum = 0, blockIdx = 0, blockLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}