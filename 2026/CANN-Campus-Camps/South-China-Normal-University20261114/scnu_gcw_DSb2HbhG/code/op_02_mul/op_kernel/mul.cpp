// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

// 每个队列申请的 buffer 数。本算子数据量很小，单 buffer 即可完成搬运与计算。
constexpr uint32_t BUFFER_NUM = 1;
// 队列深度，与 BUFFER_NUM 保持一致
constexpr uint32_t QUEUE_DEPTH = 1;
// 单个 tile 的元素个数：256 个 float 为 1024B，256 个 half 为 512B，均为 32B 的整数倍
constexpr uint32_t TILE_LENGTH = 256;
// GM <-> UB 搬运的最小对齐粒度（字节）
constexpr uint32_t BLOCK_BYTES = 32;

// z = x * y，逐元素相乘
// 切分方式：总元素按核数向上取整，再按 32B 对齐，保证每个核的起始地址都对齐；
// 核内按 TILE_LENGTH 分块，最后一块可能不满。
template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    // length: 输入 x 的元素总个数（不是字节数）
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length)
    {
        uint32_t coreNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t alignNum = BLOCK_BYTES / static_cast<uint32_t>(sizeof(DT_X));
        if (coreNum == 0) {
            coreNum = 1;
        }

        // 先按核均分（向上取整），再向上对齐到 32B
        uint32_t perCore = (length + coreNum - 1U) / coreNum;
        perCore = (perCore + alignNum - 1U) / alignNum * alignNum;
        uint32_t gmOffset = blockIdx * perCore;

        // 最后一个核（以及数据量较小时编号更大的核）只处理剩余数据
        this->blockLength = 0;
        if (gmOffset < length) {
            uint32_t remain = length - gmOffset;
            this->blockLength = remain < perCore ? remain : perCore;
        }

        // 没有分到数据时直接返回，避免后面 InitBuffer(0)
        if (this->blockLength == 0) {
            return;
        }

        // 每个核只读写属于自己的那一段 GM，偏移单位是元素个数
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + gmOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + gmOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + gmOffset, this->blockLength);

        uint32_t tileBytes = static_cast<uint32_t>(TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }
        for (uint32_t offset = 0; offset < this->blockLength; offset += TILE_LENGTH) {
            uint32_t count = this->blockLength - offset;
            if (count > TILE_LENGTH) {
                count = TILE_LENGTH;
            }
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        DataCopy(xLocal, xGm[offset], count);
        DataCopy(yLocal, yGm[offset], count);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 向量乘：z = x * y
        Mul(zLocal, xLocal, yLocal, count);

        outQueueZ.EnQue<DT_X>(zLocal);
        // 及时释放输入 buffer
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm[offset], zLocal, count);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inQueueX;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inQueueY;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

    uint32_t blockLength = 0;  // 本核实际处理的元素个数
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}