/**
 * add_custom 算子 —— Ascend C 实现
 * 功能：对两个输入张量 x, y 执行逐元素加法，z = x + y
 * 数据类型：float16   数据格式：ND   形状：2 维 [N2, N1]
 *
 * 编程范式：矢量（Vector）编程
 * 核心流程：CopyIn -> Compute -> CopyOut（双缓冲流水）
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 每块 queue 使用 2 块缓冲做双缓存

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    /**
     * 初始化：解析 tiling 参数、设置 queue 缓冲和 GlobalTensor
     * @param x 全局内存上的输入 x
     * @param y 全局内存上的输入 y
     * @param z 全局内存上的输出 z
     * @param totalLength 2 维张量展平后的总元素个数 = N2 * N1
     * @param tileLength 每次核内处理的数据长度（BLOCK 对齐）
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileLength)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();          // 每个核处理的总长度
        this->tileNum = blockLength / tileLength;                 // 每个核循环次数
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * GetBlockIdx(),
                            this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * GetBlockIdx(),
                            this->blockLength);

        // 单块 tile 的字节数 = 元素数 * sizeof(half)
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }

    // 核心执行入口：依次执行 CopyIn -> Compute -> CopyOut
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
    // 搬入：Global -> Local（VECIN）
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算：z = x + y（VECIN -> VECCALC/VECOUT）
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        Add<half>(zLocal, xLocal, yLocal, this->tileLength); // 逐元素加法指令

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 搬出：Local（VECOUT） -> Global
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;                                  // 管理内存分配的管道
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<half> xGm, yGm, zGm;

    uint32_t blockLength = 0;                    // 单核处理总长度
    uint32_t tileLength = 0;                     // 单次循环处理长度
    uint32_t tileNum = 0;                        // 循环次数
};

/**
 * 算子核函数入口（kernel 方式调用）
 * tiling 数据按 TilingData 结构体放置在 tilingAddr，由宿主侧填充：
 *   [0] totalLength  [1] tileLength
 */
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tilingAddr)
{
    KernelAddCustom op;
    __ubuf__ uint32_t *tiling = (__ubuf__ uint32_t *)get_tiling_ptr(tilingAddr);
    uint32_t totalLength = tiling[0];
    uint32_t tileLength = tiling[1];
    op.Init(x, y, z, totalLength, tileLength);
    if (ASCEND_IS_AICORE) {
        op.Process();
    }
}

#ifndef ASCENDC_CPU_DEBUG
// 核函数原型声明（用于 acl 调用）
void add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream,
                   uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *tiling)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, tiling);
}
#endif
