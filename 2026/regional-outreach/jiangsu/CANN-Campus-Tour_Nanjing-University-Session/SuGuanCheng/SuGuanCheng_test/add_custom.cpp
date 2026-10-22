/**
 * @file add_custom.cpp
 * @brief Ascend C 算子：add_custom —— 对两个 float16 张量 x, y 执行逐元素加法 z = x + y
 *
 * 形状：2 维 [N2, N1]，格式：ND，数据类型：float16
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲，搬运与计算流水并行

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    /**
     * 初始化：解析 Tiling 参数，设置流水线队列、全局内存地址
     * @param x 输入张量 x 的全局地址
     * @param y 输入张量 y 的全局地址
     * @param z 输出张量 z 的全局地址
     * @param totalLength 张量元素总数 = N2 * N1
     * @param tileLength 每次核内迭代处理（切块）的元素数
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileLength)
    {
        this->blockLength = totalLength / GetBlockNum(); // 每个 AI Core 处理的元素数
        this->tileNum = blockLength / tileLength;        // 每 Core 内的迭代次数
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ half *)x + GetBlockIdx() * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + GetBlockIdx() * this->blockLength, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + GetBlockIdx() * this->blockLength, this->blockLength);

        // Local 内存队列：每级 tile 分配 BUFFER_NUM 块缓冲
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }

    // 主执行流程：循环处理每一个 tile
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 搬入：GM -> Local（VECIN 队列）
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算：z = x + y（逐元素加法）
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

    // 搬出：Local（VECOUT 队列）-> GM
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe; // 管理内存分配的通信管道
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<half> xGm, yGm, zGm;

    uint32_t blockLength = 0; // 每个 Core 处理的元素数
    uint32_t tileNum = 0;     // 每 Core 内的迭代（切块）数
    uint32_t tileLength = 0;  // 每次迭代的元素数
};

/**
 * 核函数入口：add_custom
 * 注意：核函数必须 extern "C"，且参数中 GM_ADDR 需通过 __gm__ 指针传递
 */
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 uint32_t totalLength, uint32_t tileLength)
{
    KernelAddCustom op;
    op.Init(x, y, z, totalLength, tileLength);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// 非调试场景下，补充核函数的调用桩（供 Host 侧 icc 调用）
#endif
