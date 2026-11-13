// Kernel侧核函数实现
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

// 声明队列深度
constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {
        this->totalLength = totalLength;
        this->tileNum = tileNum;

        // 1. 获取当前运行核的编号和总核数
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        // 2. 计算 32 字节对齐的元素个数（防止搬运越界）
        uint32_t alignNum = 32 / sizeof(DT_X);
        uint32_t totalAlignLength = (totalLength + alignNum - 1) / alignNum * alignNum;

        // 3. 多核切分：计算当前核负责的数据长度和偏移量
        this->coreLength = totalAlignLength / blockNum;
        this->coreLength = (this->coreLength + alignNum - 1) / alignNum * alignNum; // 长度对齐
        this->coreOffset = blockIdx * this->coreLength;

        // 4. 安全保护：如果是最后一个核，且数据不足以填满，则截断防越界
        if (this->coreOffset >= totalAlignLength) {
            this->coreLength = 0;
        } else if (this->coreOffset + this->coreLength > totalAlignLength) {
            this->coreLength = totalAlignLength - this->coreOffset;
        }

        // 5. 计算单次 tile 处理长度
        this->tileLength = this->coreLength / tileNum;
        this->tileLength = (this->tileLength + alignNum - 1) / alignNum * alignNum;
        if (this->tileLength == 0) {
            this->tileLength = alignNum; 
        }

        this->loopCount = this->coreLength / this->tileLength;
        this->tailLength = this->coreLength % this->tileLength;

        // 6. 设置 GlobalTensor 偏移（关键：必须加 coreOffset）
        xGlobal.SetGlobalBuffer((__gm__ DT_X*)x + this->coreOffset, this->coreLength);
        yGlobal.SetGlobalBuffer((__gm__ DT_X*)y + this->coreOffset, this->coreLength);
        zGlobal.SetGlobalBuffer((__gm__ DT_X*)z + this->coreOffset, this->coreLength);

        // 7. 初始化 UB 队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        // 处理尾块数据
        if (this->tailLength > 0) {
            CopyIn(this->loopCount, this->tailLength);
            Compute(this->loopCount, this->tailLength);
            CopyOut(this->loopCount, this->tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        
        DataCopy(xLocal, xGlobal[progress * this->tileLength], length);
        DataCopy(yLocal, yGlobal[progress * this->tileLength], length);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        
        Mul(zLocal, xLocal, yLocal, length);
        
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGlobal[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGlobal, yGlobal, zGlobal;
    
    uint32_t totalLength, tileNum;
    uint32_t coreOffset, coreLength, tileLength, loopCount, tailLength;
};

// 入口函数
template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}