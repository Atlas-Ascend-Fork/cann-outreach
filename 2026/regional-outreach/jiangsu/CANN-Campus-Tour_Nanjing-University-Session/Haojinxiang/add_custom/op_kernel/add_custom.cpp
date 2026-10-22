/**
 * @file add_custom.cpp
 * @brief 算子 add_custom 的核函数 (Kernel) 实现。
 *
 * 功能: 对两个输入张量 x, y 逐元素相加, z = x + y
 * shape: 2 维 [N2, N1]   dtype: float16 (half)   format: ND
 *
 * 数据流 (向量编程范式, CopyIn -> Compute -> CopyOut 流水线 + 双缓冲):
 *   GM(x) --DataCopy--> UB(xLocal)
 *   GM(y) --DataCopy--> UB(yLocal)
 *   UB(yLocal) + UB(xLocal) --Add--> UB(zLocal)
 *   UB(zLocal) --DataCopy--> GM(z)
 */
#include "kernel_operator.h"
#include "add_custom_tiling.h"

namespace {
constexpr int32_t BUFFER_NUM = 2;  // 每个队列中的 tensor 数量 (双缓冲, 用于 DMA 与计算重叠)
}  // namespace

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    // totalLength: 总元素数 (N2*N1); tileNum: 单核切分出的 tile 数
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 每个 AI Core 处理的数据长度 (块长度)
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 单个 tile 的长度 (再按双缓冲切为两份, 供 DMA 与计算交替使用)
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        // 每个 Core 处理自己 blockLength 长度的一段, 用 GetBlockIdx() 偏移
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 为队列申请 UB 空间 (单位: 字节), 每队列 BUFFER_NUM 块, 每块 tileLength 个 half
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        // 双缓冲, 所以循环次数为 tileNum * BUFFER_NUM
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 第 progress 个 tile: 把 x, y 从 GM 搬到 UB
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算 z = x + y
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 第 progress 个 tile: 把结果 z 从 UB 搬回 GM
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // 解析 Host 侧写入的 TilingData (也可写作 GET_TILING_DATA(tilingData, tiling))
    const AddCustomTilingData &tilingData = *reinterpret_cast<const AddCustomTilingData *>(tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// 框架 (ACLNN) 侧启动核函数的包装
void add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z,
                   uint8_t *workspace, uint8_t *tiling)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif
