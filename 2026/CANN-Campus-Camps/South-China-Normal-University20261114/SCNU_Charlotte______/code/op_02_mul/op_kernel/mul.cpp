// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;   // 每个队列的 buffer 块数
constexpr int32_t QUEUE_DEPTH = 1;  // 队列深度


template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData &tiling) {
        // 每个核处理的数据长度（多核并行，按核数均分）
        this->blockLength = tiling.totalLength / AscendC::GetBlockNum();
        // 单核内分块数
        this->tileNum = tiling.tileNum;
        // 单次处理的元素数（tileLength 已按 BUFFER_NUM 折半）
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        // 设置每个核在 Global Memory 上的起始地址与长度（多核切分的关键）
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        // 循环次数 = tileNum × BUFFER_NUM
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);   // 搬入第 i 块数据
            Compute(i);  // 计算第 i 块数据
            CopyOut(i);  // 搬出第 i 块结果
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 申请 LocalTensor 内存
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        // 从 Global Memory 搬入数据到 Local Memory
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        // 放入队列，通知 Compute 任务
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列取出数据
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        // 申请输出内存
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 矢量乘法：z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
        // 放入输出队列，通知 CopyOut 任务
        outQueueZ.EnQue<DT_X>(zLocal);
        // 释放输入数据内存
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取出计算结果
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        // 从 Local Memory 搬出到 Global Memory
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        // 释放输出数据内存
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;  // TPipe 内存管理对象
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;   // 输入队列（VECIN）
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;           // 输出队列（VECOUT）
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;                                    // Global Memory 管理对象
    uint32_t blockLength = 0;  // 每核处理的元素数
    uint32_t tileNum = 0;      // 单核内分块数
    uint32_t tileLength = 0;   // 单核内每块元素数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}
