/*
 * add_custom 算子 kernel 侧实现（Ascend C）
 * 功能：z = x + y，逐元素加法
 * 数据类型：float16；数据格式：ND；shape：2 维 [N2, N1]（拉平为一维处理）
 * 流水：CopyIn -> Compute -> CopyOut，双缓冲（BUFFER_NUM = 2）
 */
#include "add_custom_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲，拷入/计算/拷出流水并行

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    // 初始化：按 AI Core 分块，设置全局内存与 Local 内存缓冲
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum(); // 每个 AI Core 处理的元素数
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM; // 单次搬运/计算长度

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x) + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y) + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(z) + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    // 主流程：tileNum * BUFFER_NUM 次流水迭代
    __aicore__ inline void Process()
    {
        const int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 搬入：Global Memory -> Local Memory
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算：z = x + y（AI Core Vector 指令）
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        Add(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 搬出：Local Memory -> Global Memory
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    uint32_t blockLength = 0; // 每个 AI Core 处理的元素数
    uint32_t tileNum = 0;     // 每核循环块数
    uint32_t tileLength = 0;  // 单块元素数
};

// 算子 kernel 入口：tiling 由 host 侧下发
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
