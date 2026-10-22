/*
 * @file add0123.cpp (op_kernel)
 * kernel侧核心实现: z = x + y
 * dtype: float16, Format: ND, shape 2维 [N2, N1] (按展平总量分核)
 * 实现: TQue 双缓冲流水线 (CopyIn -> Compute -> CopyOut)
 *       Add 向量指令整片批量计算, DataCopyPad 处理尾块, 无标量循环
 */
#include "kernel_operator.h"
#include "add0123_tiling.h"

using namespace AscendC;

constexpr uint32_t USE_CORE_NUM = 8;  // 与 op_host 侧 BLOCK_DIM 保持一致
constexpr uint32_t TILE_LENGTH = 128; // 每片元素数 (half, 单次向量 repeat = 128 元素)
constexpr uint32_t BUFFER_NUM = 2;    // 双缓冲, 搬运与计算流水并行

class KernelAdd0123 {
public:
    __aicore__ inline KernelAdd0123() {}

    // 初始化: 按核均分数据, 尾核兜底余量; 为队列申请 LocalMemory 缓冲
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockPerCore = totalLength / USE_CORE_NUM;
        uint32_t startOffset = blockIdx * blockPerCore;
        blockLength = (blockIdx == USE_CORE_NUM - 1)
                          ? (totalLength - startOffset)
                          : blockPerCore;
        loopCount = (blockLength + TILE_LENGTH - 1) / TILE_LENGTH;

        xGm.SetGlobalBuffer((__gm__ half *)x + startOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + startOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + startOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }

    // 标准三级流水
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 搬入: GM -> Local (VECIN 队列), DataCopyPad 支持任意长度尾块
    __aicore__ inline void CopyIn(int32_t progress)
    {
        uint32_t offset = progress * TILE_LENGTH;
        uint32_t len = (blockLength - offset < TILE_LENGTH) ? (blockLength - offset) : TILE_LENGTH;
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopyExtParams copyParams{1, len * sizeof(half), 0, 0, 0};
        DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算: 向量化 Add, 一次整片 128 个 half, 无标量循环;
    // 尾片未搬满部分计算的是缓冲区内无效数据, 搬出时按实际长度截断即可
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        Add(zLocal, xLocal, yLocal, TILE_LENGTH); // z = x + y

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 搬出: Local (VECOUT 队列) -> GM
    __aicore__ inline void CopyOut(int32_t progress)
    {
        uint32_t offset = progress * TILE_LENGTH;
        uint32_t len = (blockLength - offset < TILE_LENGTH) ? (blockLength - offset) : TILE_LENGTH;
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopyExtParams copyParams{1, len * sizeof(half), 0, 0, 0};
        DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    uint32_t blockLength = 0;
    uint32_t loopCount = 0;
};

// 算子入口: 标准签名 (x, y, z, workspace, tiling)
extern "C" __global__ __aicore__ void add0123(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd0123 op;
    op.Init(x, y, z, tilingData.totalLength);
    op.Process();
}
