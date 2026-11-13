// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; 

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 由于已知 shape 为 (8, 2048)，总长 16384
        // 无论分多少个核，每个核处理的数据长度绝对是 32 字节对齐的
        this->blockLength = totalLength / GetBlockNum();
        
        // 切分成 tileNum 块，每块长度 = blockLength / tileNum
        // 例如：8核，每核处理 2048 个元素。切 2 块，每块 1024 个元素。
        this->tileLength = this->blockLength / tileNum;
        this->tileNum = tileNum;

        // 计算当前核的全局偏移
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + this->blockLength * GetBlockIdx(), this->blockLength);

        // 初始化队列（直接分配精确长度，零浪费）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 循环 tileNum * BUFFER_NUM 次
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        // 使用极速的 DataCopy（因为长度完美对齐，不用 DataCopyPad）
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 直接计算，长度绝对对齐
        Mul(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();

        // 极速输出
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}