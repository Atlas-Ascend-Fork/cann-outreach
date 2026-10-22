/**
 * add_custom.cpp
 *
 * add_custom 算子核函数（Ascend C 矢量算子）。
 *
 * 功能：对两个输入张量 x、y 执行逐元素加法，输出 z = x + y。
 * 数据类型：float16（half）；数据格式：ND；shape：2 维 [N2, N1]。
 *
 * 编程范式：矢量三级流水（CopyIn -> Compute -> CopyOut）+ 双缓冲（double buffer）。
 * 数据并行：Host 侧按 numBlocks 个核切分整体数据，单核内再按 tileNum * BUFFER_NUM 个块
 *           循环搬入 / 计算 / 搬出，搬运与计算通过 Queue 重叠，实现流水线并行。
 */
#include "kernel_operator.h"
#include "add_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // 每个 Queue 中缓存的 buffer 个数（双缓冲）

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    // 初始化：计算单核 / 单块长度，设置各核 Global Memory 起始地址，并为 Queue 分配 UB 内存
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();  // 每个核负责的数据长度（元素数）
        this->tileNum = tileNum;                          // 每个核内数据切分的块数
        ASSERT(this->tileNum != 0 && "tile num can not be zero!");
        // 每个 buffer 的长度：单核数据量再按 tileNum 分块、并按双缓冲一分为二
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        // 依据当前核索引偏移 GM 起始地址，实现多核并行的数据切分
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * GetBlockIdx(), this->blockLength);

        // 为输入 / 输出 Queue 分配 UB 内存（单位：Byte），每个 Queue 分配 BUFFER_NUM 块
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    // 核心处理：循环完成「搬入 -> 计算 -> 搬出」三级流水
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;  // 双缓冲使循环次数翻倍
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 搬入：把第 progress 个块从 Global Memory 拷到 Local Memory 并放入输入 Queue
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算：从输入 Queue 取出数据，执行逐元素加法 z = x + y，结果放入输出 Queue
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        // 矢量双目加法接口
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        // 释放输入 LocalTensor，供下一轮复用
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 搬出：从输出 Queue 取出结果，写回 Global Memory
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;                                   // Pipe 内存管理对象
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;  // 输入 Queue（VECIN）
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;          // 输出 Queue（VECOUT）
    GlobalTensor<half> xGm, yGm, zGm;             // Global Memory 地址管理对象
    uint32_t blockLength;                         // 每个核负责的数据长度（元素数）
    uint32_t tileNum;                             // 每个核内数据切分的块数
    uint32_t tileLength;                          // 每个 buffer 的数据长度（元素数）
};

// 核函数入口：参数顺序固定为「输入、输出、workspace、tiling」，不可调整
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // add 为纯矢量计算，指定核类型为 AIV（Vector）
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 注册默认 TilingData 结构体，随后读取 Host 侧写入的 tiling 参数
    REGISTER_TILING_DEFAULT(AddCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
