// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

// 每个队列的buffer块数（开启double buffer，实现搬运与计算流水并行）
constexpr int32_t BUFFER_NUM = 2;
// 单次处理的最大元素个数，用于控制UB占用（按数据块分块计算）
constexpr uint32_t MAX_TILE_LENGTH = 2048;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // 多核切分：每核处理长度按32字节对齐，保证各核Global Memory起始地址满足DataCopy的对齐要求
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t alignNum = static_cast<uint32_t>(32 / sizeof(DT_X));  // 32字节对应的元素个数
        uint32_t perCoreLength = (length + blockNum - 1) / blockNum;
        perCoreLength = (perCoreLength + alignNum - 1) / alignNum * alignNum;
        // 本核数据在整块张量中的起始元素偏移
        this->blockOffset = blockIdx * perCoreLength;
        if (this->blockOffset > length) {
            this->blockOffset = length;
        }
        this->blockLength = length - this->blockOffset;
        if (this->blockLength > perCoreLength) {
            this->blockLength = perCoreLength;
        }

        // 单次搬运/计算的元素个数：按32字节对齐，且不超过UB容量限制
        uint32_t tileLength = this->blockLength;
        if (tileLength > MAX_TILE_LENGTH) {
            tileLength = MAX_TILE_LENGTH;
        }
        tileLength = (tileLength / alignNum) * alignNum;
        if (tileLength < alignNum) {
            tileLength = alignNum;
        }
        this->tileLength = tileLength;

        // 设置本核对应的Global Memory地址与长度
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + this->blockOffset, this->blockLength);
        // 为队列分配UB内存，每个队列BUFFER_NUM块
        pipe.InitBuffer(inQueueX, BUFFER_NUM, static_cast<uint32_t>(this->tileLength * sizeof(DT_X)));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, static_cast<uint32_t>(this->tileLength * sizeof(DT_X)));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, static_cast<uint32_t>(this->tileLength * sizeof(DT_X)));
    }
    __aicore__ inline void Process() {
        // 逐块处理本核数据，最后一块可能不足tileLength
        for (uint32_t offset = 0; offset < this->blockLength; offset += this->tileLength) {
            uint32_t curLength = this->blockLength - offset;
            if (curLength > this->tileLength) {
                curLength = this->tileLength;
            }
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        if (curLength * sizeof(DT_X) % 32 == 0) {
            // 长度满足32字节对齐时使用DataCopy，性能更优
            AscendC::DataCopy(xLocal, xGm[offset], curLength);
            AscendC::DataCopy(yLocal, yGm[offset], curLength);
        } else {
            // 尾块长度非32字节对齐时使用DataCopyPad，自动补齐
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
            AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t curLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 矢量乘法：z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, curLength);
        outQueueZ.EnQue<DT_X>(zLocal);
        // 释放输入数据内存
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        if (curLength * sizeof(DT_X) % 32 == 0) {
            AscendC::DataCopy(zGm[offset], zLocal, curLength);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        }
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;                                                          // TPipe内存管理对象
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;    // 输入队列，VECIN位置
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;            // 输出队列，VECOUT位置
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;                                    // 输入输出Global Memory

    uint32_t blockOffset;  // 本核在整块张量中的起始元素偏移
    uint32_t blockLength;  // 本核处理的元素个数
    uint32_t tileLength;   // 单核内每块处理的元素个数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
