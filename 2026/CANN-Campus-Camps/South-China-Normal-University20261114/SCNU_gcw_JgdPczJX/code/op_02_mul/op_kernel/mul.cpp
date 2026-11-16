#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        uint32_t coreNum = GetBlockNum();
        uint32_t coreIdx = GetBlockIdx();
        
        // 核心修改 1：保证按 32 字节对齐分配 blockLength
        uint32_t alignElements = 32 / sizeof(DT_X);
        this->blockLength = (length + coreNum - 1) / coreNum;
        this->blockLength = (this->blockLength + alignElements - 1) / alignElements * alignElements;
        
        uint32_t startOffset = coreIdx * this->blockLength;
        if (startOffset >= length) {
            this->curLength = 0;
            return;
        }
        
        // 核心修改 2：计算真实需要处理的剩余长度（防止越界）
        this->curLength = (length - startOffset < this->blockLength) ? (length - startOffset) : this->blockLength;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + startOffset, this->curLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + startOffset, this->curLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + startOffset, this->curLength);

        // 核心修改 3：取消多余的分块和双缓冲，直接分配够用的空间
        pipe.InitBuffer(inQueueX, 1, this->blockLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 1, this->blockLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 1, this->blockLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->curLength == 0) return;

        // 核心修改 4：去掉复杂的 for 循环，直接一把全搬进去算
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        DataCopy(xLocal, xGm, this->curLength);
        DataCopy(yLocal, yGm, this->curLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);

        xLocal = inQueueX.DeQue<DT_X>();
        yLocal = inQueueY.DeQue<DT_X>();

        Mul(zLocal, xLocal, yLocal, this->curLength);

        outQueueZ.EnQue<DT_X>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);

        zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm, zLocal, this->curLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t blockLength{0};
    uint32_t curLength{0};
};

// 入口函数保持不变
template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}