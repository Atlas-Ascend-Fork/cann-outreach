// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

// 单缓冲 + 单趟：整个核只有 1 次循环、2 条搬入 DMA、1 条 Mul、1 条搬出 DMA、
// 4 个同步事件——是队列框架下的最小执行序列（实测比多块流水更快）
constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t blockLength, uint32_t tileNum, uint32_t tileLength) {
        // tiling 均为 host 预计算值，核内零除法、无 GetBlockNum 调用
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        // 每个核按核编号偏移，只访问自己负责的 GM 数据段（多核切分）
        const uint64_t offset = (uint64_t)this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, this->blockLength);
        // 块字节数只算一次，两个队列复用
        const uint32_t tileBytes = this->tileLength * (uint32_t)sizeof(DT_X);
        // x / y 两路输入合并到一个输入队列、一块连续 UB 缓冲（前半 x、后半 y），
        // Alloc/EnQue/DeQue/Free 与事件同步指令减半
        pipe.InitBuffer(inQueueXY, BUFFER_NUM, 2 * tileBytes);
        // 输出队列
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileBytes);
    }

    __aicore__ inline void Process() {
        // 循环次数 = tileNum（规格输入下为 1，单趟完成）
        const int32_t loopCount = (int32_t)this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);    // 搬入第 i 块数据
            Compute(i);   // 计算第 i 块数据
            CopyOut(i);   // 搬出第 i 块结果
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 当前块在 GM 中的起始偏移（一次计算，x / y 共用）
        const uint32_t gmOffset = (uint32_t)progress * this->tileLength;
        // 一次 Alloc 同时得到 x、y 两路输入内存
        AscendC::LocalTensor<DT_X> xyLocal = inQueueXY.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> xLocal = xyLocal;
        AscendC::LocalTensor<DT_X> yLocal = xyLocal[this->tileLength];
        // 从 GM 异步搬入两路输入（同一 MTE2 队列背靠背下发，引擎内连续搬运）
        AscendC::DataCopy(xLocal, xGm[gmOffset], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[gmOffset], this->tileLength);
        // 一次 EnQue 的同步事件同时覆盖两路搬运完成
        inQueueXY.EnQue(xyLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 一次 DeQue 同时取出 x、y 两路数据
        AscendC::LocalTensor<DT_X> xyLocal = inQueueXY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> xLocal = xyLocal;
        AscendC::LocalTensor<DT_X> yLocal = xyLocal[this->tileLength];
        // 为计算结果分配内存
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 矢量乘法：z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
        // 结果入队，一次 FreeTensor 同时释放 x、y 的输入内存
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueXY.FreeTensor(xyLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 当前块的输出偏移与 CopyIn 一致
        const uint32_t gmOffset = (uint32_t)progress * this->tileLength;
        // 取出结果，从 UB 搬出到 GM 对应位置，释放输出内存
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[gmOffset], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;                                                // UB 内存管理对象
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueXY;   // 合并后的输入队列（x、y 共用）
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;  // 输出队列
    AscendC::GlobalTensor<DT_X> xGm;  // 输入 x 的 GM 管理对象
    AscendC::GlobalTensor<DT_X> yGm;  // 输入 y 的 GM 管理对象
    AscendC::GlobalTensor<DT_X> zGm;  // 输出 z 的 GM 管理对象
    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 每块元素数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    // tiling 三项均为 host 预计算值，直接传入
    op.Init(x, y, z, tiling_data.blockLength, tiling_data.tileNum, tiling_data.tileLength);
    op.Process();
}