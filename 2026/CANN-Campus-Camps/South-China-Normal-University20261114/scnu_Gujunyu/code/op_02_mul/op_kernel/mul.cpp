// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;   // double buffer
constexpr int32_t TILE_NUM   = 8;   // 单核内再切分的块数

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length)
    {
        // 32B 对齐所需元素个数：fp32 -> 8，fp16 -> 16
        uint32_t alignNum = 32 / sizeof(DT_X);

        uint32_t coreNum = AscendC::GetBlockNum();   // host 侧 SetBlockDim 的核数
        uint32_t coreIdx = AscendC::GetBlockIdx();   // 当前核号

        // 每核长度先按 32B 向下对齐，除不尽的部分交给最后一个核
        uint32_t alignedBlock = length / coreNum;
        alignedBlock = alignedBlock / alignNum * alignNum;
        if (alignedBlock == 0) {
            alignedBlock = alignNum;
        }

        uint32_t start = alignedBlock * coreIdx;
        if (start >= length) {
            blockLength = 0;
            tileLength = alignNum;
            return;
        }

        uint32_t remain = length - start;
        blockLength = (coreIdx == coreNum - 1) ? remain : alignedBlock;
        if (blockLength > remain) {
            blockLength = remain;
        }

        // 单核内再切 TILE_NUM × BUFFER_NUM 块，块长同样按 32B 对齐
        tileLength = blockLength / (TILE_NUM * BUFFER_NUM);
        tileLength = tileLength / alignNum * alignNum;
        if (tileLength == 0) {
            tileLength = alignNum;
        }

        // 按核号把 GM 指针平移到本核区间
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + start, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + start, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + start, blockLength);

        // 为三个队列分配 UB 内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        while (offset < blockLength) {
            uint32_t count = blockLength - offset;
            if (count > tileLength) {
                count = tileLength;      // 最后一块可能不满，按实际剩余长度处理
            }
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], count);
        AscendC::DataCopy(yLocal, yGm[offset], count);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 矢量乘法：z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, count);

        outQueueZ.EnQue<DT_X>(zLocal);
        // 输入 buffer 必须释放，否则下一轮 AllocTensor 拿不到 UB 空间
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, count);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t blockLength;   // 本核处理的总元素数
    uint32_t tileLength;    // 单块元素数
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
