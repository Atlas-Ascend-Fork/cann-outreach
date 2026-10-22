/*
 * add_custom 算子：z = x + y（逐元素加法）
 * 输入/输出均为 float16，ND 格式，shape 为 [N2, N1]（2 维）
 * 数据在 GM 中按 [N2, N1] 连续排布，核函数侧按一维连续内存做逐元素加法
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;   // 双缓冲，搬运与计算流水并行
constexpr int32_t TILE_ALIGN = 32;  // half 对齐粒度（32 个半字 = 64 字节）

/* tiling 结构：host 侧计算后传给核函数，需与 host 侧保持一致 */
struct AddCustomTilingData {
    uint32_t totalLength;   // N2 * N1，总元素个数
    uint32_t tileLength;    // 每次搬运/计算的长度（TILE_ALIGN 对齐）
    uint32_t tilingKey;     // 切分策略编号
};

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                const AddCustomTilingData& tiling)
    {
        this->tileLength = tiling.tileLength;
        // 每核均分（向上取整），最后一个核取实际剩余量，避免越界
        uint32_t blockNum = GetBlockNum();
        uint32_t perCore = (tiling.totalLength + blockNum - 1) / blockNum;
        uint32_t start = GetBlockIdx() * perCore;
        this->blockLength = (start + perCore > tiling.totalLength)
                                ? (tiling.totalLength - start)
                                : perCore;
        this->tileNum = blockLength / this->tileLength;  // 每核整块循环次数
        this->lastLength = blockLength % this->tileLength;  // 每核尾部残量

        // 各核按自己的 [start, start + blockLength) 切分 GM 区间
        xGm.SetGlobalBuffer((__gm__ half*)x + start, blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + start, blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + start, blockLength);

        // 统一缓冲区：每段 tileLength * sizeof(half) 字节
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = static_cast<int32_t>(this->tileNum);
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        // 处理每核尾部非整块残量（按 TILE_ALIGN 向上取整，尾部不足部分用 0 补齐计算后写回残量长度）
        if (this->lastLength > 0) {
            CopyInTail();
            ComputeTail();
            CopyOutTail();
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);   // z = x + y
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

    // ---- 尾部残量处理 ----
    __aicore__ inline uint32_t TailAlignedLen()
    {
        return ((this->lastLength + TILE_ALIGN - 1) / TILE_ALIGN) * TILE_ALIGN;
    }

    __aicore__ inline void CopyInTail()
    {
        uint32_t len = TailAlignedLen();
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopyPad(xLocal, xGm[this->tileNum * this->tileLength],
                    len, this->lastLength);
        DataCopyPad(yLocal, yGm[this->tileNum * this->tileLength],
                    len, this->lastLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void ComputeTail()
    {
        uint32_t len = TailAlignedLen();
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutTail()
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[this->tileNum * this->tileLength], zLocal, this->lastLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;                                   // 管理统一内存
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;

    uint32_t blockLength = 0;   // 每核处理的数据个数
    uint32_t tileLength = 0;    // 每次搬运/计算长度
    uint32_t tileNum = 0;       // 每核整块循环次数
    uint32_t lastLength = 0;    // 每核尾部残量
};

/* 核函数入口（NPU 侧） */
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelAddCustom op;
    op.Init(x, y, z, tilingData);
    op.Process();
}
