// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;   // 每个队列申请的 buffer 块数（开启 double buffer）
constexpr uint32_t QUEUE_DEPTH = 1;  // 队列深度

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    // 初始化：多核切分 + 单核内切块 + UB 内存分配
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // 每个核处理的数据长度（多核并行）
        this->blockLength = length / AscendC::GetBlockNum();
        // 单核内数据切块数，需保证整除且搬运长度满足 32 字节对齐
        this->tileNum = 8;
        while (this->tileNum > 1 &&
               (this->blockLength % (this->tileNum * BUFFER_NUM) != 0 ||
                this->blockLength / (this->tileNum * BUFFER_NUM) < 32 / sizeof(DT_X))) {
            this->tileNum /= 2;
        }
        // 单核内每块长度，再按 BUFFER_NUM 折半以支持双层流水
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        // 设置每个核的 Global Memory 起始地址（多核切分逻辑）
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    // 核心流程：CopyIn -> Compute -> CopyOut 三级流水
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 搬入：Global Memory -> Local Memory
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算：Local Memory 上执行矢量乘法
    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 搬出：Local Memory -> Global Memory
    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;  // TPipe 内存管理对象
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;  // 输入队列（VECIN）
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;          // 输出队列（VECOUT）
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;                                  // 全局内存地址管理对象

    uint32_t blockLength;  // 每核处理的元素数
    uint32_t tileNum;      // 单核内分块数
    uint32_t tileLength;   // 单核内每块元素数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}